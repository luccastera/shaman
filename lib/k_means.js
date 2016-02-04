/*globals require, exports */

var distance = require('./distance.js').euclidean,
    _ = require('underscore');

var KMeans = function(K, options) {
  options = options || {};
  this.K = K || 3;
  this.centroids = [];
  this.clusters = [];
  this.iterations = options.iterations || 1000;

  this.options = options || {};
}

KMeans.prototype.cluster = function(data, callback) {
  var self = this;

  if (!data) {
    return callback(new Error('data is required.'));
  } else if (!Array.isArray(data)) {
    return callback(new Error('data must be an array.'));
  } else if (data.length < self.K) {
    return callback(new Error('data must have at least K data points.'));
  }

  var normalizedData = self.normalize(data);

  // initialize random centroids
  for (var k = 0; k < self.K; k++) {
    var randomIndex = Math.floor(Math.random() * normalizedData.length);
    self.centroids.push(normalizedData[randomIndex]);
  }

  for (var j = 0; j < self.iterations; j++) {

    // cluster assignment step
    var clusterIndexes = [];
    for (var i = 0; i < normalizedData.length; i++) {
      var min = distance(normalizedData[i], self.centroids[0]);
      var closestCentroid = 0;
      for (k = 0; k < self.K; k++) {
        var tmpDistance = distance(normalizedData[i], self.centroids[k]);
        if (tmpDistance < min) {
          min = tmpDistance;
          closestCentroid = k;
        }
      }
      clusterIndexes.push(closestCentroid);
    }

    // move centroids
    var newCentroids = [];
    self.clusters = [];
    for (var k = 0; k < self.centroids.length; k++) {
      var points = [];
      clusterIndexes.forEach(function(clusterIndex, index) {
        if (clusterIndex == k) {
          points.push(normalizedData[index]);
        }
      });
      self.clusters.push(points);
      if (points.length > 0) {
        newCentroids.push( self.mean(points) );
      } else {
        newCentroids.push(self.centroids[k]);
      }
    }
    self.centroids = newCentroids;
  }

  // denormalize clusters and centroids
  self.centroids = self.denormalize(self.centroids);
  self.clusters.forEach(function(cluster, index) {
    self.clusters[index] = self.denormalize(cluster);
  });

  return callback(null, self.clusters, self.centroids);
};

KMeans.prototype.mean = function(points) {
  if (!Array.isArray(points)) {
    throw new Error('mean requires an array of data points as an argument.');
  }
  if (points.length === 0) {
    return [];
  }
  var sum = new Array(points[0].length);
  for (var i = 0; i < points.length; i++) {
    for (var k = 0; k < sum.length; k++) {
      sum[k] = (sum[k] || 0) + parseFloat(points[i][k], 10);

      if (i === points.length - 1) {
        sum[k] = sum[k] / points.length;
      }

    }
  }
  return sum;
};

KMeans.prototype.normalize = function(points) {
  var mean = this.mean(points);
  this.originalMean = mean;
  var newPoints = [];
  points.forEach(function(point, j) {
    var newPoint = new Array(point.length);
    for (var i = 0; i < point.length; i++) {
      newPoint[i] = (point[i] - mean[i]) / mean[i];
    }
    newPoints.push(newPoint);
  });
  return newPoints;
};

KMeans.prototype.denormalize = function(points) {
  var originalMean = this.originalMean;
  var newPoints = [];
  points.forEach(function(point, j) {
    var newPoint = new Array(point.length);
    for (var i = 0; i < point.length; i++) {
      newPoint[i] = (point[i] * originalMean[i]) + originalMean[i];
    }
    newPoints.push(newPoint);
  });
  return newPoints;
};

exports.KMeans = KMeans;


