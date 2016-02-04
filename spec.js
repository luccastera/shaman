/*globals describe, require */

var LinearRegression = require('./index').LinearRegression,
    euclideanDistance = require('./index').euclideanDistance,
    KMeans = require('./index').KMeans,
    assert = require('assert'),
    _ = require('underscore'),
    sinon = require('sinon');

var fixtures = {};

var EPSILON = 0.01; // used to compare floats

describe('LinearRegresssion', function() {
  describe('initialization', function() {
    it('can be initialized with no parameters', function(done) {
      var lr = new LinearRegression();
      assert.ok(lr);
      done();
    });

    it('should throw an error if X is not an array', function(done) {
      var x = 'a string';
      assert.throws(function() {
        var lr = new LinearRegression(x);
      }, Error);
      done();
    });

    it('should throw an error if Y is not an array', function(done) {
      var x = [1];
      var y = 'a string';
      assert.throws(function() {
        var lr = new LinearRegression(x, y);
      }, Error);
      done();
    });
  });

  describe('addColumnOne', function() {
    it('should append a column on all 1 in front of the matrix', function(done) {
      var X = [[4,5], [6,7]];
      var newX = LinearRegression.addColumnOne(X);
      assert.equal(1, newX.e(1,1));
      assert.equal(1, newX.e(2,1));
      assert.equal(4, newX.e(1,2));
      assert.equal(5, newX.e(1,3));
      assert.equal(6, newX.e(2,2));
      assert.equal(7, newX.e(2,3));
      done();
    });
  });

  describe('normalize', function() {
    it('should normalize features correctly', function(done) {
      var features = [[10, 2], [15,2], [20, 2]];
      var lr = new LinearRegression(features, [2,4,5]);
      var X = LinearRegression.addColumnOne(features);
      var normalizedX = lr.normalize(X);
      assert.equal(1, normalizedX.e(1,1));
      assert.equal(1, normalizedX.e(2,1));
      assert.equal(1, normalizedX.e(3,1));
      assert.equal(-0.5, normalizedX.e(1,2));
      assert.equal(0, normalizedX.e(2,2));
      assert.equal(0.5, normalizedX.e(3,2));
      assert.equal(-0, normalizedX.e(1,3));
      assert.equal(0, normalizedX.e(2,3));
      assert.equal(0, normalizedX.e(3,3));
      done();
    });
  });

  describe('train', function() {
    it('should throw an error if there is no data in X', function(done) {
      var lr = new LinearRegression();
      lr.train(function(err) {
        assert.ok(err);
        assert.equal(err.message, 'X is empty');
        done();
      });
    });
    it('should throw an error if there is no data in Y', function(done) {
      var lr = new LinearRegression([0,1,2,3]);
      lr.train(function(err) {
        assert.ok(err);
        assert.equal(err.message, 'Y is empty');
        done();
      });
    });
    it('should throw an error if X and Y have different length', function(done) {
      var lr = new LinearRegression([0,1,2], [0]);
      lr.train(function(err) {
        assert.ok(err);
        assert.equal(err.message, 'X and Y must be of the same length');
        done();
      });
    });
    it('should throw an error on vertical line since we cannot inverse matrix', function(done) {
      var lr = new LinearRegression([0, 0], [1,0]);
      lr.train(function(err) {
        assert.ok(err);
        assert.equal(err.message, 'could not inverse the matrix in normal equation. Try to use Gradient Descent instead.');
        done();
      });
    });
    it('should not save cost function if we are using Normal Equation', function(done) {
      var lr = new LinearRegression([0,1,2,3,4,5,6,7], [0,2,4,3,7,6,8,9], {algorithm: 'NormalEquation'});
      lr.train(function(err) {
        assert.ok(!err);
        assert.ok(!lr.costs);
        done();
      });
    });
    it('should save cost function if using Gradient Descent and asked to do so', function(done) {
      var lr = new LinearRegression([0,1,2,3,4,5,6,7], [0,2,4,3,7,6,8,9], {
        algorithm: 'GradientDescent',
        saveCosts: true,
        learningRate: 0.1,
        numberOfIterations: 100
      });
      lr.train(function(err) {
        assert.ok(!err);
        assert.ok(lr.costs);
        assert.equal(lr.costs.length, 100);
        done();
      });
    });
  });

  describe('predict', function() {
    it('should throw an error if called before training', function(done) {
      var lr = new LinearRegression([0,1,2,3,4], [1,3,4,5,6]);
      assert.throws(function() {
        lr.predict(1);
      }, Error);
      done();
    });
  });

  function predictionTests() {
    describe('simple linear regression', function() {
      it('should correctly generates a line for a 0,0 to 1,1 dataset (slope of 1)', function(done) {
        var lr = new LinearRegression([0, 1], [0,1], fixtures.options);
        lr.train(function(err) {
          assert.ok(lr.predict(0) - 0 < EPSILON);
          assert.ok(lr.predict(0.5) -  0.5 < EPSILON);
          assert.ok(lr.predict(1) - 1 < EPSILON);
          done();
        });
      });
      it('should correctly generates a line for a (0,0) to (1,0) dataset (horizontal line)', function(done) {
        var lr = new LinearRegression([0, 1], [0,0], fixtures.options);
        lr.train(function(err) {
          assert.equal(lr.predict(0), 0);
          assert.equal(lr.predict(0.5), 0);
          assert.equal(lr.predict(1), 0);
          done();
        });
      });
      it('should correctly generates a line for a (0,5) to (1,5) dataset (horizontal line)', function(done) {
        var lr = new LinearRegression([0, 1], [5,5], fixtures.options);
        lr.train(function(err) {
          assert.ok(lr.predict(0) - 5 < EPSILON);
          assert.ok(lr.predict(0.5) - 5 < EPSILON);
          assert.ok(lr.predict(1) - 5 < EPSILON);
          done();
        });
      });
      it('should handle single point input of (0,0)', function(done) {
        var lr = new LinearRegression([0], [0], fixtures.options);
        lr.train(function(err) {
          assert.equal(lr.predict(10), 0);
          done();
        });
      });
      it('should handle a single point example by returning y-intercept', function(done) {
        var lr = new LinearRegression([0], [1], fixtures.options);
        lr.train(function(err) {
          assert.equal(lr.predict(5), 5);
          done();
        });
      });
      it('should predict a simple example correctly', function(done) {
        var lr = new LinearRegression([1, 2, 3, 4, 5], [2, 2, 3, 3, 5], fixtures.options);
        lr.train(function(err) {
          assert.ok(lr.predict(0) - 0.899 < EPSILON);
          assert.ok(lr.predict(1) - 1.599 < EPSILON);
          assert.ok(lr.predict(2) - 2.3 < EPSILON);
          assert.ok(lr.predict(3) - 2.999 < EPSILON);
          assert.ok(lr.predict(4) - 3.699 < EPSILON);
          assert.ok(lr.predict(5) - 4.4 < EPSILON);
          assert.ok(lr.predict(10) - 7.9 < EPSILON);
          done();
        });
      });
    });
    describe('multiple linear regression', function() {
      it('should predict a simple example correctly');
    });
  }

  describe('train and predict', function() {
    it('should train with Normal Equation by default', function(done) {
      var lr = new LinearRegression([1, 2, 3, 4, 5], [2, 2, 3, 3, 5]);
      assert.ok(lr.algorithm, 'NormalEquation');
      var spy = sinon.spy(lr, 'trainWithNormalEquation');
      lr.train(function(err) {
        assert.ok(spy.called);
        done();
      });
    });
    it('should train with Normal Equation if asked to do so', function(done) {
      var lr = new LinearRegression([1, 2, 3, 4, 5], [2, 2, 3, 3, 5], {algorithm: 'NormalEquation'});
      assert.ok(lr.algorithm, 'NormalEquation');
      var spy = sinon.spy(lr, 'trainWithNormalEquation');
      lr.train(function(err) {
        assert.ok(spy.called);
        done();
      });
    });
    it('should train with GradientDescent if asked to do so', function(done) {
      var lr = new LinearRegression([1, 2, 3, 4, 5], [2, 2, 3, 3, 5], {algorithm: 'GradientDescent'});
      assert.ok(lr.algorithm, 'GradientDescent');
      var spy = sinon.spy(lr, 'trainWithGradientDescent');
      lr.train(function(err) {
        assert.ok(spy.called);
        done();
      });
    });
    describe('with Normal Equation', function() {
      predictionTests();
    });
    describe('with Gradient Descent', function() {
      beforeEach(function(callback) {
        fixtures.options = {algorithm: 'GradientDescent'};
        return callback();
      });
      predictionTests();
      describe('multiple linear regression', function() {
        it('should correctly generates a line for a [0,0] -> 0 and [1,1]  -> 1 dataset', function(done) {
          var lr = new LinearRegression([[0,0], [1,1]], [0,1], fixtures.options);
          lr.train(function(err) {
            assert.ok(!err, err);
            assert.ok(lr.predict([0,0]) - 0 < EPSILON);
            assert.ok(lr.predict([0.5,0.5]) -  0.5 < EPSILON);
            assert.ok(lr.predict([1,1]) - 1 < EPSILON);
            done();
          });
        });
      });
    });
  });
});

describe('Euclidean Distance', function() {
  it('the euclidean distance between two identical vectors should be zero', function(done) {
    var a = [5,5];
    var b = [5,5];
    assert.equal(euclideanDistance(a,b), 0);
    done();
  });
  it('should calculate the euclidean distance between two vectors of size 2', function(done) {
    var a = [5,5];
    var b = [0,0];
    assert.equal(euclideanDistance(a,b), 7.0710678118654755);
    done();
  });
});

describe('k-means', function() {
  describe('initialization', function() {
    it('can be initialized with no parameters - K defaults to 3', function(done) {
      var kmeans = new KMeans();
      assert.ok(kmeans);
      assert.equal(kmeans.K, 3);
      done();
    });
    it('can be initialize with a value for K', function(done) {
      var kmeans = new KMeans(5);
      assert.ok(kmeans);
      assert.equal(kmeans.K, 5);
      done();
    });
  });

  describe('mean', function() {
    it('should error if no data points is passed', function(done) {
      var kmeans = new KMeans();
      assert.throws(function() {
        var average = kmeans.mean();
      }, Error);
      done();
    });
    it('should return an empty array if passed an empty array as input', function(done) {
      var kmeans = new KMeans();
      var average = kmeans.mean([]);
      assert.deepEqual(average, []);
      done();
    });
    it('should return the mean data point', function(done) {
      var kmeans = new KMeans();
      var data = [[0,0], [1,1], [2,2]];
      assert.deepEqual(kmeans.mean(data), [1,1]);
      done();
    });

    it('should return the mean data point', function(done) {
      var kmeans = new KMeans();
      var data = [[0,0,1], [1,1,1], [2,2,5], [10,15,2], [5,8,4.5]];
      assert.deepEqual(kmeans.mean(data), [3.6,5.2,2.7]);
      done();
    });

  });

  describe('cluster', function() {
    it('should error if no data is passed', function(done) {
      var kmeans = new KMeans();
      kmeans.cluster(null, function(err, clusters) {
        assert.ok(err);
        assert.equal(err.message, 'data is required.');
        done();
      });
    });
    it('should error if data is not an array', function(done) {
      var kmeans = new KMeans();
      kmeans.cluster({}, function(err, clusters) {
        assert.ok(err);
        assert.equal(err.message, 'data must be an array.');
        done();
      });
    });
    it('should error if we do not have enough  data points (m < K)', function(done) {
      var kmeans = new KMeans(3);
      var data = [[1,1], [2,2]];
      kmeans.cluster(data, function(err, clusters) {
        assert.ok(err);
        assert.equal(err.message, 'data must have at least K data points.');
        done();
      });
    });
    it('should return an array of clusters', function(done) {
      var kmeans = new KMeans(3);
      var data = [[1,1], [2,1], [4,5], [6,7]];
      kmeans.cluster(data, function(err, clusters) {
        assert.ok(!err);
        assert.ok(clusters);
        assert.ok(Array.isArray(clusters));
        done();
      });
    });

    it('should return an array of centroids of size K', function(done) {
      var kmeans = new KMeans(3);
      var data = [[1,1], [2,1], [4,5], [6,7]];
      kmeans.cluster(data, function(err, clusters, centroids) {
        assert.ok(!err);
        assert.ok(centroids);
        assert.ok(Array.isArray(centroids));
        assert.equal(centroids.length, 3);
        done();
      });
    });

    it('should cluster correctly into 2 clusters', function(done) {
      var kmeans = new KMeans(2);
      var data = [
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
        [20, 20, 20],
        [200, 200, 200]
      ];
      kmeans.cluster(data, function(err, clusters) {
        assert.ok(!err);
        assert.ok(clusters.length === 2);
        assert.ok(_.contains(clusters.map(function(c) { return c.join(','); }), [[200,200,200]].map(function(c) { return c.join(','); }).join(',')));
        assert.ok(_.contains(clusters.map(function(c) { return c.map(function(i) { return i.map(function(j) { return Math.floor(j);}); }).join(','); }), [[1, 1, 1],[ 2, 2, 2],[3, 3, 3],[4, 4, 4],[5, 5, 5],[20, 20, 20]].map(function(c) { return c.join(','); }).join(',')));
        done();
      });
    });

    it('should cluster correctly into 3 clusters', function(done) {
      var kmeans = new KMeans(3);
      var data = [
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
        [20, 20, 20],
        [40, 40, 40],
        [200, 200, 200]
      ];
      kmeans.cluster(data, function(err, clusters, centroids) {
        assert.ok(!err);
        assert.ok(clusters.length === 3);
        assert.ok(_.contains(clusters.map(function(c) { return c.join(','); }), [[200,200,200]].map(function(c) { return c.join(','); }).join(',')));
        assert.ok(_.contains(clusters.map(function(c) { return c.join(','); }), [[20,20,20], [40,40,40]].map(function(c) { return c.join(','); }).join(',')));
        assert.ok(_.contains(clusters.map(function(c) { return c.join(','); }), [[1, 1, 1],[ 2, 2, 2],[3, 3, 3],[4, 4, 4],[5, 5, 5]].map(function(c) { return c.join(','); }).join(',')));
        assert.ok(_.contains(centroids.map(function(c) { return c.join(','); }), '3,3,3'));
        done();
      });
    });

  });
});
