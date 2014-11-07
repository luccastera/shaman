# shaman

Machine Learning library for node.js

## Linear Regression

Linear Regression is implemented using linear algebra and the normal equation.

### Usage

By default, shaman uses the Normal Equation for linear regression.

```javascript
var X = [1, 2, 3, 4, 5];
var Y = [2, 2, 3, 3, 5];
var lr = new LinearRegression(X,Y);
lr.train(function(err) {
  if (err) { throw err; }
  
  // you can now start using lr.predict:
  console.log(lr.predict(1));
});
```

If your data does not work well with the Normal Equation, you can also
use the Gradient Descent algorithm as an alternative.

```javascript
var X = [1, 2, 3, 4, 5];
var Y = [2, 2, 3, 3, 5];
var lr = new LinearRegression(X,Y, {
  algorithm: 'GradientDescent'
});
lr.train(function(err) {
  if (err) { throw err; }
  
  // you can now start using lr.predict:
  console.log(lr.predict(1));
});
```

When using Gradient Descent, you can define the number of iterations
(numberOfIterations and the learning rate (learningRate) as options to
the LinearRegression function.


```javascript
var lr = new LinearRegression(X,Y, {
  algorithm: 'GradientDescent',
  numberOfIterations: 1000, // defaults to 8500
  learningRate: 0.5 // defaults to 0.1
});
```

When using the Gradient Descent algorithm, you can ask shaman to save
the results of the cost function at each iteration of the algorithm.
This can be useful if you would like to plot the cost function to ensure
that it is converging.

```javascript
var lr = new LinearRegression(X,Y, {
  algorithm: 'GradientDescent',
  saveCosts: true // defaults to false
});
lr.train(function(err) {
  // you can now get they array of costs:
  console.log(lr.costs);
});
```


See how to use with Gradient Descent algorithm [here](examples/usage-gd.js).

### Examples

[Click here](https://plot.ly/~luccastera/2) to see an example of Simple Linear Regression
using the Normal Equation to evaluate the price of cars based on their horsepower that was done with the shaman
library. Code is in [examples/cars.js](examples/cars.js)).

[Click here](https://plot.ly/~luccastera/3/aapl-stock-prices/) to see an
example of Simple Linear Regression applies to the stock price of AAPL
using the Gradient Descent algorithm from 2008 to 2012. Code can be seen at
[examples/stock.js](examples/stock.js).

[Click here](https://plot.ly/~luccastera/4/cigarettes/) to see an
exmaple of Multiple Linear Regression to evaluate Carbon Monoxide in
cigarettes from nicotine and tar content. Code can be seen at
[examples/cigarettes.js](examples/cigarettes.js).


### License

[MIT](LICENSE)
