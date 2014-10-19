/*globals describe, require */

var LinearRegression = require('./index').LinearRegression,
    assert = require('assert');

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
  });

  describe('predict', function() {
    it('should predict a simple example correctly', function(done) {
      var lr = new LinearRegression([1, 2, 3, 4, 5], [2, 2, 3, 3, 5]);
      lr.train(function(err) {
        assert.ok(lr.predict(0) - 0.899 < 0.01);
        assert.ok(lr.predict(1) - 1.599 < 0.01);
        assert.ok(lr.predict(2) - 2.3 < 0.01);
        assert.ok(lr.predict(3) - 2.999 < 0.01);
        assert.ok(lr.predict(4) - 3.699 < 0.01);
        assert.ok(lr.predict(5) - 4.4 < 0.01);
        assert.ok(lr.predict(10) - 7.9 < 0.01);
        done();
      });
    });
  });
});
