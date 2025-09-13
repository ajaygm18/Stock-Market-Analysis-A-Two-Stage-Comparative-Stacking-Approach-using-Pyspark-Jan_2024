<?php
if (isset($_POST['runPrediction'])) {
    // Get input values from the form
    $open = $_POST['open'];
    $high = $_POST['high'];
    $low = $_POST['low'];

    // Validate or sanitize input values if needed

    // Define the Python script path
    $pythonScript = "finalStockPrediction.py"; // Update with the correct path

    // Build the command to execute the Python script with input values
    $command = "python $pythonScript $open $high $low";

    // Execute the command and capture the output
    $output = shell_exec($command);

    // Print the output for debugging (you can remove this line in a production environment)
    echo "<pre>$output</pre>";

    // Send the predicted value back to the frontend
    echo "<script>document.getElementById('predictedValue').innerHTML = 'Predicted Value: $output';</script>";
}
?>
