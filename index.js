/*globals require */

var sylvester = require('sylvester'),
    Matrix = sylvester.Matrix,
    Vector = sylvester.Vector,
    _ = require('underscore');

var LinearRegression = function(X, Y, options) {
  this.X = X || [];
  this.Y = Y || [];
  this.options = options || {};
  this.trained = false;

  if (this.options.algorithm === 'GradientDescent') {
    this.algorithm = 'GradientDescent';
  } else if (this.options.algorithm === 'NormalEquation') {
    this.algorithm = 'NormalEquation';
  } else {
    this.algorithm = 'NormalEquation';
  }

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

  // verify that X and Y inputs have the same length
  if (this.X.length !== this.Y.length) {
    return callback(new Error('X and Y must be of the same length'));
  }

  // if there is only one point, let's just choose a
  // slope of 0 and a y-intercept of the y passed in
  if (this.X.length === 1) {
    this.theta = $M([0, this.Y[0]]);
    this.trained = true;
    return callback();
  }

  if (this.algorithm === 'GradientDescent') {
    return this.trainWithGradientDescent(callback);
  } else {
    return this.trainWithNormalEquation(callback);
  }
};

LinearRegression.addColumnOne = function(X) {
  // The x matrix for the normal equation needs to
  // have a row of ones as its first row.
  // Let's first build the x matrix
  var zeros = Matrix.Zero(X.length,1);
  var ones = zeros.add(1);
  var x = ones.augment($M(X));
  return x;
};

LinearRegression.prototype.trainWithNormalEquation = function(callback) {
  var x = LinearRegression.addColumnOne(this.X);
  // Build the y matrix
  var y = $M(this.Y);

  // now we can apply the normal equation:
  // see formula at http://upload.wikimedia.org/math/2/c/e/2ce21b8e24ea7509a3295c3acd2ae0ea.png
  var inversePortion = x.transpose().x(x).inverse();
  if (inversePortion) {
    this.theta = inversePortion.x(x.transpose()).x(y);
    this.trained = true;
    return callback(); 
  } else {
    return callback(new Error('could not inverse the matrix in normal equation'));
  }
};

LinearRegression.computeCost = function(X, Y, theta) {
  var m = Y.dimensions().rows;
  var xThetaMinusYSquared = (X.x(theta)).subtract(Y).map(function(val) { return val * val; });
  var xThetaMinusYArray = _.flatten(xThetaMinusYSquared.elements);
  var sum = _.reduce(xThetaMinusYArray, function(memo, num) { return memo + num; }, 0);
  return (1 / (2 * m)) * sum;
};

LinearRegression.gradientDescent = function(X, Y, theta, learningRate, numberOfIterations) {
  var m = Y.dimensions().rows;
  for (var i = 0; i < numberOfIterations; i++) {
    var xThetaMinusY = (X.x(theta)).subtract(Y);

    var sum1Array = _.flatten(xThetaMinusY.elements);
    var sum1 = _.reduce(sum1Array, function(memo, num) { return memo + num; }, 0);

    var sum2Matrix = xThetaMinusY.transpose().x(X);
    var sum2Array = _.flatten(sum2Matrix.elements);
    var sum2 = _.reduce(sum2Array, function(memo, num) { return memo + num; }, 0);

    var temp1 = theta.e(1,1) - (learningRate / m) * sum1;
    var temp2 = theta.e(2,1) - (learningRate / m) * sum2;
    theta = $M([[temp1], [temp2]]);
    //console.log('cost', LinearRegression.computeCost(X, Y, theta));
  }
  return theta;
};

LinearRegression.prototype.trainWithGradientDescent = function(callback) {
  var learningRate = this.options.learningRate || 0.1;
  var numberOfIterations = this.options.numberOfIterations || 8500;

  // initialize theta to zero
  this.theta = Matrix.Zero(2, 1);

  var x = LinearRegression.addColumnOne(this.X);

  // Build the y matrix
  var y = $M(this.Y);

  var cost = LinearRegression.computeCost(x, y, this.theta);

  this.theta = LinearRegression.gradientDescent(x, y, this.theta, learningRate, numberOfIterations);
  this.trained = true;
  return callback(); 
};

LinearRegression.prototype.predict = function(input) {
  if (this.trained) {
    var xInput = $M([1, input]);
    var output = this.theta.transpose().x(xInput);
    return output.e(1,1);
  } else {
    throw new Error('cannot predict before training');
  }
};

exports.LinearRegression = LinearRegression;
