// The data used for this example was pulled from
// http://archive.ics.uci.edu/ml/datasets/Automobile

var csvParse = require('csv-parse'),
    fs = require('fs'),
    LinearRegression = require('../index').LinearRegression,
    _ = require('underscore');

var apiKey = process.env.PLOTLY_API_KEY;
var username = process.env.PLOTLY_USERNAME;
var plotly = require('plotly')(username,apiKey);

fs.readFile('./examples/cigarettes.dat', 'utf8', function(err, dataStr) {
  if (err) {
    console.log(err);
    process.exit(1);
  }
  csvParse(dataStr, {delimiter: ',', auto_parse: true}, function(err, data) {

    var X = data.map(function(r) { return [Number(r[1]), Number(r[2])]; });
    var y = data.map(function(r) { return Number(r[4]); });

    // Initialize and train the linear regression
    var lr = new LinearRegression(X, y, {algorithm: 'NormalEquation'});
    lr.train(function(err) {
      if (err) {
        console.log('error', err);
        process.exit(2);
      }

      // Use the linear regression function to get a set of data to graph the linear regression line
      var y2 = [];
      X.forEach(function(xi) {
        y2.push(lr.predict(xi));
      });

      // let's create vars to plot
      var x1 = X.map(function(item) { return item[0]; });
      var x2 = X.map(function(item) { return item[1]; });

      // Create scatter plots of training data + linear regression function
      var layout = {
        title: 'Cigarettes',
        xaxis: {
          title: 'Tar'
        },
        yaxis: {
          title: 'Nicotine'
        },
        zaxis: {
          title: 'Carbon Monoxide'
        }
      };
      var trace1 = {
        x: x1,
        y: y,
        z: x2,
        name: 'Training Data',
        mode: "markers",
        type: "scatter3d"
      };
      var trace2 = {
        x: x1,
        y: y2,
        z: x2,
        name: 'Linear Regression',
        mode: "surface",
        type: "scatter3d"
      };
      var plotData = [trace1, trace2];
      var graphOptions = {layout: layout, filename: "cigarettes-linear-regression-with-shaman", fileopt: "overwrite"}
      plotly.plot(plotData, graphOptions, function (err, msg) {
        if (err) {
          console.log(err);
          process.exit(3);
        } else {
          console.log('Success! The plot (' + msg.filename + ') can be found at ' + msg.url);
          process.exit();
        }
      });
    });
  });
});


