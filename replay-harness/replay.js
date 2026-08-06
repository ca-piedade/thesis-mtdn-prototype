'use strict';

/**
 * Replay harness for the ICDLT 2026 paper prototype.
 *
 * Reads events.json (produced by generate_events.py, same scenario/seed as
 * simulation.py) and submits each event as a REAL transaction against the
 * deployed 'fnbtrust' chaincode on the Fabric test-network, timing every
 * call. This replaces the paper's third-party-benchmark comparison (Table
 * IV vs. [6]) with actual measured latency/throughput on our own chaincode.
 *
 * v2: submits transactions with bounded CONCURRENCY instead of one at a
 * time. Waiting for full commit before sending the next transaction badly
 * underestimates achievable throughput (most of the ~2s per call is Fabric's
 * orderer batch-cutting timeout, not real work, and it amortizes across many
 * in-flight transactions) — this is also how the throughput figures in [6]
 * that Table IV compares against were themselves measured, so this is the
 * fair comparison, not the serial one.
 *
 * Expected layout (copy this whole replay-harness/ folder to be a SIBLING of
 * test-network/, i.e. into ~/fabric/fabric-samples/replay-harness/):
 *
 *   fabric-samples/
 *     test-network/          <- already deployed, channel 'mychannel', cc 'fnbtrust'
 *     replay-harness/
 *       replay.js            <- this file
 *       package.json
 *       events.json          <- copy generate_events.py's output here
 *
 * Run:  node replay.js [path/to/events.json] [--limit N] [--concurrency C]
 *   --limit N        only replay the first N events (default: all)
 *   --concurrency C  number of transactions in flight at once (default: 15)
 */

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const grpc = require('@grpc/grpc-js');
const { connect, signers } = require('@hyperledger/fabric-gateway');

const CHANNEL_NAME = process.env.CHANNEL_NAME || 'mychannel';
const CHAINCODE_NAME = process.env.CHAINCODE_NAME || 'fnbtrust';
const MSP_ID = process.env.MSP_ID || 'Org1MSP';
const PEER_ENDPOINT = process.env.PEER_ENDPOINT || 'localhost:7051';
const PEER_HOST_ALIAS = process.env.PEER_HOST_ALIAS || 'peer0.org1.example.com';

const CRYPTO_PATH =
  process.env.CRYPTO_PATH ||
  path.resolve(__dirname, '..', 'test-network', 'organizations', 'peerOrganizations', 'org1.example.com');
const TLS_CERT_PATH = path.resolve(CRYPTO_PATH, 'peers', 'peer0.org1.example.com', 'tls', 'ca.crt');
const CERT_DIR = path.resolve(CRYPTO_PATH, 'users', 'User1@org1.example.com', 'msp', 'signcerts');
const KEY_DIR = path.resolve(CRYPTO_PATH, 'users', 'User1@org1.example.com', 'msp', 'keystore');

// --- CLI args ---------------------------------------------------------------
const rawArgs = process.argv.slice(2);
// Indices that belong to a --flag's VALUE must be excluded from the
// positional-argument list, otherwise e.g. "--concurrency 15" leaks "15" in
// as if it were the events.json path.
const consumedIndices = new Set();
for (let idx = 0; idx < rawArgs.length; idx++) {
  if (rawArgs[idx].startsWith('--')) consumedIndices.add(idx + 1);
}
const positional = rawArgs.filter((a, idx) => !a.startsWith('--') && !consumedIndices.has(idx));
function flagValue(name, def) {
  const idx = rawArgs.indexOf(`--${name}`);
  if (idx === -1 || idx + 1 >= rawArgs.length) return def;
  return Number(rawArgs[idx + 1]);
}
const EVENTS_PATH = positional[0] || path.resolve(__dirname, 'events.json');
const LIMIT = flagValue('limit', Infinity);
const CONCURRENCY = flagValue('concurrency', 15);
const RESULTS_PATH = path.resolve(__dirname, 'results.json');
const PROGRESS_EVERY = 100;

function firstFileIn(dirPath) {
  const files = fs.readdirSync(dirPath);
  if (files.length === 0) throw new Error(`No files found in ${dirPath}`);
  return path.join(dirPath, files[0]);
}

async function newGrpcConnection() {
  const tlsRootCert = fs.readFileSync(TLS_CERT_PATH);
  const tlsCredentials = grpc.credentials.createSsl(tlsRootCert);
  return new grpc.Client(PEER_ENDPOINT, tlsCredentials, {
    'grpc.ssl_target_name_override': PEER_HOST_ALIAS,
    'grpc.max_send_message_length': -1,
    'grpc.max_receive_message_length': -1,
  });
}

function newIdentity() {
  const certPath = firstFileIn(CERT_DIR);
  return { mspId: MSP_ID, credentials: fs.readFileSync(certPath) };
}

function newSigner() {
  const keyPath = firstFileIn(KEY_DIR);
  const privateKey = crypto.createPrivateKey(fs.readFileSync(keyPath));
  return signers.newPrivateKeySigner(privateKey);
}

async function main() {
  if (!fs.existsSync(EVENTS_PATH)) {
    console.error(`events.json not found at ${EVENTS_PATH}`);
    process.exit(1);
  }
  let events = JSON.parse(fs.readFileSync(EVENTS_PATH, 'utf8'));
  if (Number.isFinite(LIMIT)) events = events.slice(0, LIMIT);
  console.log(`Loaded ${events.length} events (concurrency=${CONCURRENCY}) from ${EVENTS_PATH}`);

  // Group by `key` (lotId / articleId / inventory period+section). Events
  // sharing a key touch the same ledger entity and MUST run in the original
  // order; different keys are independent and safe to run concurrently.
  const groupsMap = new Map();
  for (const ev of events) {
    if (!groupsMap.has(ev.key)) groupsMap.set(ev.key, []);
    groupsMap.get(ev.key).push(ev);
  }
  const groups = Array.from(groupsMap.values());
  console.log(`Grouped into ${groups.length} independent ordering-keys`);

  const client = await newGrpcConnection();
  const gateway = connect({
    client,
    identity: newIdentity(),
    signer: newSigner(),
    evaluateOptions: () => ({ deadline: Date.now() + 5000 }),
    endorseOptions: () => ({ deadline: Date.now() + 15000 }),
    submitOptions: () => ({ deadline: Date.now() + 5000 }),
    commitStatusOptions: () => ({ deadline: Date.now() + 60000 }),
  });

  const results = [];
  let okCount = 0;
  let errCount = 0;
  let nextGroup = 0;
  let completed = 0;
  const wallStart = Date.now();
  const network = gateway.getNetwork(CHANNEL_NAME);

  async function submitOne(ev) {
    const contract = network.getContract(CHAINCODE_NAME, ev.contract);
    const t0 = process.hrtime.bigint();
    let ok = true;
    let errMsg = null;
    try {
      await contract.submitTransaction(ev.function, ...ev.args);
    } catch (err) {
      ok = false;
      errMsg = String(err.message || err).slice(0, 200);
    }
    const t1 = process.hrtime.bigint();
    const latencyMs = Number(t1 - t0) / 1e6;
    if (ok) okCount++; else errCount++;
    results.push({ key: ev.key, contract: ev.contract, function: ev.function, latencyMs, ok, errMsg });
    completed++;
    if (completed % PROGRESS_EVERY === 0 || completed === events.length) {
      const elapsedS = (Date.now() - wallStart) / 1000;
      const avgLat = results.reduce((a, r) => a + r.latencyMs, 0) / results.length;
      console.log(
        `[${completed}/${events.length}] elapsed=${elapsedS.toFixed(1)}s ` +
        `ok=${okCount} err=${errCount} avgLatency=${avgLat.toFixed(1)}ms ` +
        `throughputSoFar=${(completed / elapsedS).toFixed(2)}tx/s`
      );
    }
  }

  // Bounded-concurrency worker pool over GROUPS (not raw events): each worker
  // claims a whole group (e.g. one lot's full lifecycle) and submits that
  // group's events strictly in order, one at a time, before claiming the
  // next group. Different workers process different groups concurrently, so
  // there are ~CONCURRENCY independent lots/articles in flight at once,
  // without ever racing two events that touch the same ledger key.
  async function worker() {
    while (true) {
      const g = nextGroup++;
      if (g >= groups.length) return;
      for (const ev of groups[g]) {
        await submitOne(ev);
      }
    }
  }

  try {
    const workers = Array.from({ length: CONCURRENCY }, () => worker());
    await Promise.all(workers);
  } finally {
    gateway.close();
    client.close();
  }

  const wallEndS = (Date.now() - wallStart) / 1000;
  const latencies = results.map((r) => r.latencyMs);
  const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
  const sorted = [...latencies].sort((a, b) => a - b);
  const p50 = sorted[Math.floor(sorted.length * 0.5)];
  const p95 = sorted[Math.floor(sorted.length * 0.95)];
  const p99 = sorted[Math.floor(sorted.length * 0.99)];
  const throughputTxPerSec = results.length / wallEndS;

  const summary = {
    totalTransactions: results.length,
    concurrency: CONCURRENCY,
    ok: okCount,
    errors: errCount,
    wallClockSeconds: wallEndS,
    measuredThroughputTxPerSec: throughputTxPerSec,
    avgLatencyMs: avgLatency,
    p50LatencyMs: p50,
    p95LatencyMs: p95,
    p99LatencyMs: p99,
  };

  console.log('\n=== SUMMARY ===');
  console.log(JSON.stringify(summary, null, 2));
  fs.writeFileSync(RESULTS_PATH, JSON.stringify({ summary, results }, null, 1));
  console.log(`\nFull results written to ${RESULTS_PATH}`);
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
