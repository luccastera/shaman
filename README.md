# shaman

Machine Learning library for node.js

## Linear Regression

Linear Regression is implemented using linear algebra and the normal equation.

### Usage

```javascript
var X = [1, 2, 3, 4, 5];
var Y = [2, 2, 3, 3, 5];
var lr = new LinearRegression(X,Y);
lr.train(function(err) {
  if (err) { throw err; }
  
  // you can now start using lr.predict:
  console.log(lr.predict(1));
});

### Example

[Click here](https://plot.ly/~luccastera/2) to see an example of Simple Linear Regression
to evaluate the price of cars based on their horsepower that was done with the shaman library.
