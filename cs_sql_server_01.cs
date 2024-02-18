using System;
using System.Data.SqlClient;

namespace EmployeeDataRetrieval
{
    class Program
    {
        static void Main(string[] args)
        {
            // Replace with your actual connection string
            string connectionString = "Data Source=myServerAddress;Initial Catalog=myDatabase;User ID=myUsername;Password=myPassword";

            // Replace with the actual RESTful API endpoint
            string apiEndpoint = "https://api.example.com/employees/{employeeId}";

            // Replace with the employeeId you want to retrieve
            int employeeId = 123;

            try
            {
                using (SqlConnection connection = new SqlConnection(connectionString))
                {
                    connection.Open();

                    // Create a SQL command to retrieve employee data
                    string sqlQuery = $"SELECT * FROM employees WHERE employeeId = {employeeId}";
                    using (SqlCommand command = new SqlCommand(sqlQuery, connection))
                    {
                        SqlDataReader reader = command.ExecuteReader();
                        if (reader.HasRows)
                        {
                            while (reader.Read())
                            {
                                // Retrieve and display employee data
                                int empId = reader.GetInt32(0);
                                string empName = reader.GetString(1);
                                // Add other relevant fields here

                                Console.WriteLine($"Employee ID: {empId}");
                                Console.WriteLine($"Employee Name: {empName}");
                                // Display other relevant fields

                                // You can process the retrieved data further as needed
                            }
                        }
                        else
                        {
                            Console.WriteLine("No employee found with the specified ID.");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }
    }
}
