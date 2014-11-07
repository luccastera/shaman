// the data used for this example was borrowed from the following site:
// http://bl.ocks.org/tmcw/3931800/1bed6c10905952539000301086378c6ed1c75c84

var csvParse = require('csv-parse'),
    fs = require('fs'),
    LinearRegression = require('../index').LinearRegression,
    _ = require('underscore');

var apiKey = process.env.PLOTLY_API_KEY;
var username = process.env.PLOTLY_USERNAME;
var plotly = require('plotly')(username,apiKey);

fs.readFile('./examples/stock.tsv', 'utf8', function(err, dataStr) {
  if (err) {
    console.log(err);
    process.exit(1);
  }
  csvParse(dataStr, {delimiter: '\t', auto_parse: true}, function(err, data) {
    // We are only going two columns:
    //     x: Date
    //     y: AAPL Stock Price
    var x = data.map(function(h) {
      return Date.parse(h[0]);
    });
    var y = data.map(function(h) { return h[1]; }); // y is price

    // for the plot, we need the x values in a format that plotly can understand
    var xForPlot = _.map(data, function(d) {
      var date = new Date(Date.parse(d[0]));
      return date.toISOString().slice(0,10);
    });


    // Initialize and train the linear regression
    var lr = new LinearRegression(x, y, {algorithm: 'GradientDescent', learningRate: 0.1, numberOfIterations: 5000});
    lr.train(function(err) {
      if (err) {
        console.log('error', err);
        process.exit(2);
      }

      // Use the linear regression function to get a set of data to graph the linear regression line
      var y2 = [];
      x.forEach(function(xi) {
        y2.push(lr.predict(xi));
      });

      // Create scatter plots of training data + linear regression function
      var layout = {
        title: 'AAPL Stock Prices',
        xaxis: {
          title: 'Date'
        },
        yaxis: {
          title: 'Price in $'
        }
      };
      var trace1 = {
        x: xForPlot,
        y: y,
        name: 'Training Data',
        mode: "lines",
        type: "scatter"
      };
      var trace2 = {
        x: xForPlot,
        y: y2,
        name: 'Linear Regression',
        mode: "lines",
        type: "scatter"
      };
      var plotData = [trace1, trace2];
      var graphOptions = {layout: layout,filename: "aapl-stock-linear-regression-with-shaman", fileopt: "overwrite"}
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

