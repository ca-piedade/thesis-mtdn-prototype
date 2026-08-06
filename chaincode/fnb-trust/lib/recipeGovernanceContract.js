'use strict';

const { Contract } = require('fabric-contract-api');

// State machine (paper Section 3.2):
// Submitted -> Validated -> Active -> Superseded, exception -> Flagged

class RecipeGovernanceContract extends Contract {

  constructor() {
    super('RecipeGovernanceContract');
  }

  async _getFt(ctx, ftId) {
    const data = await ctx.stub.getState(ftId);
    if (!data || data.length === 0) {
      throw new Error(`Recipe ${ftId} does not exist`);
    }
    return JSON.parse(data.toString());
  }

  async _putFt(ctx, ft) {
    await ctx.stub.putState(ft.ftId, Buffer.from(JSON.stringify(ft)));
  }

  // Deterministic timestamp — see StockLotContract._txTimeIso for why
  // new Date()/Date.now() must never be used inside chaincode.
  _txTimeIso(ctx) {
    const ts = ctx.stub.getTxTimestamp();
    const millis = ts.seconds.toNumber() * 1000 + Math.floor(ts.nanos / 1e6);
    return new Date(millis).toISOString();
  }

  // ftSubmit(ftId, articleId, version, authorPseudonym) -> Submitted
  async ftSubmit(ctx, ftId, articleId, version, authorPseudonym) {
    const existing = await ctx.stub.getState(ftId);
    if (existing && existing.length > 0) {
      throw new Error(`Recipe ${ftId} already submitted`);
    }
    const ft = {
      docType: 'ficha_tecnica',
      ftId,
      articleId,
      version,
      authorPseudonym,
      state: 'Submitted',
      submittedAt: this._txTimeIso(ctx),
    };
    await this._putFt(ctx, ft);
    return JSON.stringify(ft);
  }

  // ftValidate(ftId, validatorPseudonym, approvalHash) -> Validated
  async ftValidate(ctx, ftId, validatorPseudonym, approvalHash) {
    const ft = await this._getFt(ctx, ftId);
    if (ft.state !== 'Submitted') {
      throw new Error(`Recipe ${ftId} must be Submitted to ftValidate (was ${ft.state})`);
    }
    ft.validatorPseudonym = validatorPseudonym;
    ft.approvalHash = approvalHash;
    ft.state = 'Validated';
    await this._putFt(ctx, ft);
    return JSON.stringify(ft);
  }

  // ftActivate(ftId, articleId, effectiveDate) -> Active
  async ftActivate(ctx, ftId, articleId, effectiveDate) {
    const ft = await this._getFt(ctx, ftId);
    if (ft.state !== 'Validated') {
      throw new Error(`Recipe ${ftId} must be Validated to ftActivate (was ${ft.state})`);
    }
    ft.articleId = articleId;
    ft.effectiveDate = effectiveDate;
    ft.state = 'Active';
    await this._putFt(ctx, ft);
    return JSON.stringify(ft);
  }

  // ftFlagged(ftId, articleId, evidenceHash, flagType) -> Flagged
  async ftFlagged(ctx, ftId, articleId, evidenceHash, flagType) {
    const ft = await this._getFt(ctx, ftId);
    if (ft.state !== 'Active') {
      throw new Error(`Recipe ${ftId} must be Active to ftFlagged (was ${ft.state})`);
    }
    ft.evidenceHash = evidenceHash;
    ft.flagType = flagType;
    ft.state = 'Flagged';
    ft.flaggedAt = this._txTimeIso(ctx);
    await this._putFt(ctx, ft);
    return JSON.stringify(ft);
  }

  // ftSupersede(ftId, newFtId, timestamp) -> Superseded
  async ftSupersede(ctx, ftId, newFtId, timestamp) {
    const ft = await this._getFt(ctx, ftId);
    if (ft.state !== 'Active' && ft.state !== 'Flagged') {
      throw new Error(`Recipe ${ftId} must be Active or Flagged to ftSupersede (was ${ft.state})`);
    }
    ft.supersededBy = newFtId;
    ft.supersededAt = timestamp;
    ft.state = 'Superseded';
    await this._putFt(ctx, ft);
    return JSON.stringify(ft);
  }

  // queryFtHistory(ftId) -> read-only, no-consensus peer query
  async queryFtHistory(ctx, ftId) {
    const iterator = await ctx.stub.getHistoryForKey(ftId);
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

module.exports = RecipeGovernanceContract;
