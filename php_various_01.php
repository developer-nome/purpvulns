<?php

echo "Hello World!";

$user_input = $_GET['input'];
echo "User input: " . $user_input;

$sql = "SELECT * FROM users WHERE username = '$user_input'";
// Execute the query

echo "File Tests!";

$file = $_GET['file'];
include($file);

$url = $_GET['url'];
include($url);

?>
