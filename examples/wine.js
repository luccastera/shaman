var csvParse = require('csv-parse'),
    fs = require('fs'),
    KMeans = require('../index').KMeans,
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

    var xAndY = data.map(function(h) { return [h[1], h[9]]; });
    var x = data.map(function(h) { return h[1]; }); // x is Malic Acid
    var y = data.map(function(h) { return h[9]; }); // y is color intensity

    var kmeans = new KMeans(3);
    kmeans.cluster(xAndY, function(err, clusters, centroids) {

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

      var centroidTrace = {
        x: centroids.map(function(c) { return c[0]; }),
        y: centroids.map(function(c) { return c[1]; }),
        mode: 'markers',
        type: 'scatter',
        name: 'Centroids',
        marker: {
          color: '#000000',
          symbol: 'cross',
          size: 10
        }
      }

      var plotData = [centroidTrace];

      clusters.forEach(function(cluster, index) {
        var trace = {
          x: cluster.map(function(c) { return c[0]; }),
          y: cluster.map(function(c) { return c[1]; }),
          mode: 'markers',
          type: 'scatter',
          name: 'Cluster ' + index
        }
        plotData.push(trace);
      });

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

});
