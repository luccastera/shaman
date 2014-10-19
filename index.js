/*globals require */

var sylvester = require('sylvester'),
    Matrix = sylvester.Matrix,
    Vector = sylvester.Vector;

var LinearRegression = function(X, Y, options) {
  this.X = X || [];
  this.Y = Y || [];
  this.options = options || {};

  // verify that X is an array
  if (X && !Array.isArray(X)) {
    throw new Error('X must be an array');
  }

  // verify that Y is an array
  if (Y && !Array.isArray(Y)) {
    throw new Error('Y must be an array');
  }
};

LinearRegression.prototype.train = function(callback) {
  if (this.X.length === 0) {
    return callback(new Error('X is empty'));
  } else if (this.Y.length === 0) {
    return callback(new Error('Y is empty'));
  }

  // normal equation using sylvester
  var xWithOnes = [];
  this.X.forEach(function(xi) {
    xWithOnes.push([1, xi]);
  });
  var x = $M(xWithOnes);
  var y = $M(this.Y);
  this.theta = x.transpose().x(x).inverse().x(x.transpose()).x(y);
  return callback();
};

LinearRegression.prototype.predict = function(input) {
  var xInput = $M([1, input]);
  var output = this.theta.transpose().x(xInput);
  return output.e(1,1);
};

exports.LinearRegression = LinearRegression;
