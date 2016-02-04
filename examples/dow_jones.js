var csvParse = require('csv-parse'),
    fs = require('fs'),
    KMeans = require('../index').KMeans,
    _ = require('underscore');

var apiKey = process.env.PLOTLY_API_KEY;
var username = process.env.PLOTLY_USERNAME;
var plotly = require('plotly')(username,apiKey);

fs.readFile('./examples/dow_jones.data', 'utf8', function(err, dataStr) {
  if (err) {
    console.log(err);
    process.exit(1);
  }

  csvParse(dataStr, {delimiter: ',', auto_parse: true}, function(err, data) {

    // let's clean up the data to keep only data for the 1/7/2011 date.
    var cleanData = _.filter(data, function(d) { return d[2] === '1/7/2011'; });

    console.log('Clustering the ' + cleanData.length + ' companies in the Down Jones Index...');


    var input = cleanData.map(function(h) {
      return [
        parseFloat(h[7], 10),
        parseFloat(h[3].slice(1), 10),
        parseFloat(h[15], 10)
      ];
    });

    var kmeans = new KMeans(3, {iterations: 10000});
    kmeans.cluster(input, function(err, clusters, centroids) {

      // plot x-y-z scatter plot in 3d
      var layout = {
        title: 'Dow Jones',
        xaxis: {
          title: 'Volume'
        },
        yaxis: {
          title: 'Stock Price'
        },
        zaxis: {
          title: 'Dividend (%)'
        }
      };

      var centroidTrace = {
        x: centroids.map(function(c) { return c[0]; }),
        y: centroids.map(function(c) { return c[1]; }),
        z: centroids.map(function(c) { return c[2]; }),
        mode: 'markers',
        type: 'scatter3d',
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
          z: cluster.map(function(c) { return c[2]; }),
          mode: 'markers',
          type: 'scatter3d',
          name: 'Cluster ' + index
        }
        plotData.push(trace);
      });

      var graphOptions = {layout: layout, filename: 'dow-jones-clustering', fileopt: 'overwrite'};
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
