var csvParse = require('csv-parse'),
    fs = require('fs'),
    kMeans = require('../index').kMeans,
    _ = require('underscore');

var apiKey = process.env.PLOTLY_API_KEY;
var username = process.env.PLOTLY_USERNAME;
var plotly = require('plotly')(username,apiKey);

fs.readFile('./examples/wine.data', 'utf8', function(err, dataStr) {
  if (err) {
    console.log(err);
    process.exit(1);
  }

  csvParse(dataStr, {delimiter: ',', auto_parse: true}, function(err, data) {

    var x = data.map(function(h) { return h[1]; }); // x is Malic Acid
    var y = data.map(function(h) { return h[9]; }); // y is color intensity

    // plot x-y scatter plot
    var layout = {
      title: 'Wine',
      xaxis: {
        title: 'Malic Acid'
      },
      yaxis: {
        title: 'Magnesium'
      }
    };

    var trace1 = {
      x: x,
      y: y,
      mode: 'markers',
      type: 'scatter'
    };
    var plotData = [trace1];
    var graphOptions = {layout: layout, filename: 'wine-categories', fileopt: 'overwrite'};
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
