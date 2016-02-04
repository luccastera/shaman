/*globals require, exports */

var distance = require('./distance.js').euclidean,
    _ = require('underscore');

var KMeans = function(K, options) {
  this.K = K || 3;
  this.centroids = [];
  this.clusters = [];

  this.options = options || {};
}

KMeans.prototype.cluster = function(data, callback) {

  if (!data) {
    return callback(new Error('data is required.'));
  } else if (!Array.isArray(data)) {
    return callback(new Error('data must be an array.'));
  } else if (data.length < this.K) {
    return callback(new Error('data must have at least K data points.'));
  }

  //var normalizedData = this.normalize(data);
  var normalizedData = data;

  // initialize random centroids
  for (var k = 0; k < this.K; k++) {
    var randomIndex = Math.floor(Math.random() * normalizedData.length);
    this.centroids.push(normalizedData[randomIndex]);
  }

  // try 100 times for now, hoping it converges
  for (var j = 0; j < 1000; j++) {

    // cluster assignment step
    var clusterIndexes = [];
    for (var i = 0; i < normalizedData.length; i++) {
      var min = distance(normalizedData[i], this.centroids[0]);
      var closestCentroid = 0;
      for (k = 0; k < this.K; k++) {
        var tmpDistance = distance(normalizedData[i], this.centroids[k]);
        if (tmpDistance < min) {
          min = tmpDistance;
          closestCentroid = k;
        }
      }
      clusterIndexes.push(closestCentroid);
    }

    // move centroids
    var newCentroids = [];
    this.clusters = [];
    for (var k = 0; k < this.centroids.length; k++) {
      var points = [];
      clusterIndexes.forEach(function(clusterIndex, index) {
        if (clusterIndex == k) {
          points.push(data[index]);
        }
      });
      this.clusters.push(points);
      if (points.length > 0) {
        newCentroids.push( this.mean(points) );
      } else {
        newCentroids.push(this.centroids[k]);
      }
    }
    this.centroids = newCentroids;
  }

  // order clusters with bigger clusters first
  //this.clusters = _.sortBy(this.clusters, function(c) { return -c.length; });

  return callback(null, this.clusters, this.centroids);
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
  var newPoints = [];
  points.forEach(function(point) {
    var newPoint = new Array(point.length);
    for (var i = 0; i < point.length; i++) {
      newPoint[i] = point[i] - mean[i];
    }
    newPoints.push(newPoint);
  });
  return newPoints;
};


exports.KMeans = KMeans;


