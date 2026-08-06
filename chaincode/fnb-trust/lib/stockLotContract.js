'use strict';

const { Contract } = require('fabric-contract-api');

// State machine (paper Section 3.1):
// Registered -> Verified -> Received -> Allocated -> Consumed, exception -> Blocked

class StockLotContract extends Contract {

  constructor() {
    super('StockLotContract');
  }

  async _getLot(ctx, lotId) {
    const data = await ctx.stub.getState(lotId);
    if (!data || data.length === 0) {
      throw new Error(`Lot ${lotId} does not exist`);
    }
    return JSON.parse(data.toString());
  }

  async _putLot(ctx, lot) {
    await ctx.stub.putState(lot.lotId, Buffer.from(JSON.stringify(lot)));
  }

  // Deterministic timestamp: derived from the transaction proposal, identical
  // on every endorsing peer. NEVER use new Date()/Date.now() in chaincode —
  // each peer would compute a different value, the resulting read-write sets
  // would not match across endorsers, and every transaction would fail with
  // "failed to collect enough endorsements".
  _txTimeIso(ctx) {
    const ts = ctx.stub.getTxTimestamp();
    const millis = ts.seconds.toNumber() * 1000 + Math.floor(ts.nanos / 1e6);
    return new Date(millis).toISOString();
  }

  // stockRegister(lotId, supplierId, poRef) -> Registered
  async stockRegister(ctx, lotId, supplierId, poRef) {
    const existing = await ctx.stub.getState(lotId);
    if (existing && existing.length > 0) {
      throw new Error(`Lot ${lotId} already registered`);
    }
    const lot = {
      docType: 'lot',
      lotId,
      supplierId,
      poRef,
      state: 'Registered',
      createdAt: this._txTimeIso(ctx),
    };
    await this._putLot(ctx, lot);
    return JSON.stringify(lot);
  }

  // validateReceipt(lotId, invoiceRef) -> Verified
  async validateReceipt(ctx, lotId, invoiceRef) {
    const lot = await this._getLot(ctx, lotId);
    if (lot.state !== 'Registered') {
      throw new Error(`Lot ${lotId} must be Registered to validateReceipt (was ${lot.state})`);
    }
    lot.invoiceRef = invoiceRef;
    lot.state = 'Verified';
    await this._putLot(ctx, lot);
    return JSON.stringify(lot);
  }

  // confirmReception(lotId, node, timestamp) -> Received
  async confirmReception(ctx, lotId, node, timestamp) {
    const lot = await this._getLot(ctx, lotId);
    if (lot.state !== 'Verified') {
      throw new Error(`Lot ${lotId} must be Verified to confirmReception (was ${lot.state})`);
    }
    lot.node = node;
    lot.receivedAt = timestamp;
    lot.state = 'Received';
    await this._putLot(ctx, lot);
    return JSON.stringify(lot);
  }

  // allocateStock(lotId, sectionNode, requisitionId) -> Allocated
  async allocateStock(ctx, lotId, sectionNode, requisitionId) {
    const lot = await this._getLot(ctx, lotId);
    if (lot.state !== 'Received') {
      throw new Error(`Lot ${lotId} must be Received to allocateStock (was ${lot.state})`);
    }
    lot.sectionNode = sectionNode;
    lot.requisitionId = requisitionId;
    lot.state = 'Allocated';
    await this._putLot(ctx, lot);
    return JSON.stringify(lot);
  }

  // recordConsumption(lotId, ftId, posRef, timestamp) -> Consumed
  // ftId references the recipe version in effect (links to RecipeGovernanceContract)
  async recordConsumption(ctx, lotId, ftId, posRef, timestamp) {
    const lot = await this._getLot(ctx, lotId);
    // Allocated: first draw against this lot. Consumed: additional draws against an
    // already-allocated lot before month-end reconciliation (a lot is typically drawn
    // on by multiple service events, not just one). Blocked: retry after correction.
    if (lot.state !== 'Allocated' && lot.state !== 'Consumed' && lot.state !== 'Blocked') {
      throw new Error(`Lot ${lotId} must be Allocated, Consumed, or Blocked to recordConsumption (was ${lot.state})`);
    }
    lot.ftId = ftId;
    lot.posRef = posRef;
    lot.consumedAt = timestamp;
    lot.state = 'Consumed';
    await this._putLot(ctx, lot);
    return JSON.stringify(lot);
  }

  // anomalyDetected(lotId, shapScore, anomalyType) -> Blocked
  // anomalyType: "consumption" | "dataQuality"
  async anomalyDetected(ctx, lotId, shapScore, anomalyType) {
    const lot = await this._getLot(ctx, lotId);
    lot.shapScore = shapScore;
    lot.anomalyType = anomalyType;
    lot.state = 'Blocked';
    lot.blockedAt = this._txTimeIso(ctx);
    await this._putLot(ctx, lot);
    return JSON.stringify(lot);
  }

  // correctionRecorded(lotId, decision, actorPseudonym, timestamp) -> back to Consumed
  async correctionRecorded(ctx, lotId, decision, actorPseudonym, timestamp) {
    const lot = await this._getLot(ctx, lotId);
    if (lot.state !== 'Blocked') {
      throw new Error(`Lot ${lotId} must be Blocked to correctionRecorded (was ${lot.state})`);
    }
    lot.decision = decision;
    lot.actorPseudonym = actorPseudonym;
    lot.correctedAt = timestamp;
    lot.state = 'Consumed';
    await this._putLot(ctx, lot);
    return JSON.stringify(lot);
  }

  // inventoryFinalized(period, section, validatedHash) -> period/section-level record, not lot-specific
  async inventoryFinalized(ctx, period, section, validatedHash) {
    const key = `INV_${period}_${section}`;
    const record = {
      docType: 'inventoryFinalization',
      period,
      section,
      validatedHash,
      finalizedAt: this._txTimeIso(ctx),
    };
    await ctx.stub.putState(key, Buffer.from(JSON.stringify(record)));
    return JSON.stringify(record);
  }

  // queryLotHistory(lotId) -> read-only, no-consensus peer query
  async queryLotHistory(ctx, lotId) {
    const iterator = await ctx.stub.getHistoryForKey(lotId);
    const history = [];
    let res = await iterator.next();
    while (!res.done) {
      if (res.value) {
        history.push({
          txId: res.value.txId,
          timestamp: res.value.timestamp,
          isDelete: res.value.isDelete,
          value: res.value.value && res.value.value.length ? JSON.parse(res.value.value.toString()) : null,
        });
      }
      res = await iterator.next();
    }
    await iterator.close();
    return JSON.stringify(history);
  }
}

module.exports = StockLotContract;
