var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  // meant to invoke a finding from nodejsscan
  var employeeId = req.foo;
  var sql = "SELECT * FROM trn_employee WHERE employee_id = " + employeeId;

  // meant to invoke a finding from nodejsscan
  res.set('X-XSS-Protection', '0')

  // meant to invoke a finding from secret finders (truffleHog, GitLeaks, etc.)
  var strAwsSecretKey = "7JyrNOrk23B7bErD88eg8IfhYjAYdFJlhCbKEo6A";

  res.render('index', { title: 'Expresssssss' });
});

module.exports = router;
