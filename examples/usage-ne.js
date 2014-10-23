var LinearRegression = require('shaman').LinearRegression;

var x = [0,1,2,3,4,5,6,7];
var y = [5,6,7,3,5,7,2,8];

var lr = new LinearRegression(x, y);

lr.train(function(err) {
  // now you can start using for prediction
  lr.predict(0);
  lr.predict(1);
});
