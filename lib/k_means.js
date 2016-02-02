/*globals require, exports */



var KMeans = function(K, options) {
  this.K = K || 3;

  this.options = options || {};
}

KMeans.prototype.cluster = function(data, callback) {

  if (!data) {
    return callback(new Error('data is required.'));
  } else if (!Array.isArray(data)) {
    return callback(new Error('data must be an array.'));
  }

  return callback(null, []);
};

exports.KMeans = KMeans;
