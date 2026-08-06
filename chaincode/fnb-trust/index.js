'use strict';

const StockLotContract = require('./lib/stockLotContract');
const RecipeGovernanceContract = require('./lib/recipeGovernanceContract');

module.exports.StockLotContract = StockLotContract;
module.exports.RecipeGovernanceContract = RecipeGovernanceContract;
module.exports.contracts = [StockLotContract, RecipeGovernanceContract];
