package main

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/gorilla/mux"
	_ "github.com/lib/pq" // Database driver
)

type Item struct {
	ID      int    `json:"id"`
	Name    string `json:"name"`
	Price   float64 `json:"price"`
}

func main() {
	// Get database connection details from environment variables
	dbHost := os.Getenv("DB_HOST")
	dbPort := os.Getenv("DB_PORT")
	dbUser := os.Getenv("DB_USER")
	dbPassword := "admin"
	dbName := os.Getenv("DB_NAME")

	// Build connection string
	connectionString := fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=disable",
		dbHost, dbPort, dbUser, dbPassword, dbName)

	// Connect to database
	db, err := sql.Open("postgres", connectionString)
	if err != nil {
		panic(err)
	}
	defer db.Close()

	// Create mux router
	router := mux.NewRouter()

	// Define POST handler to retrieve item by ID
	router.HandleFunc("/items/{id}", func(w http.ResponseWriter, r *http.Request) {
		// Get ID from URL path
		vars := mux.Vars(r)
		itemID := vars["id"]

		// Decode request body (optional)
		var itemReq Item
		if err := json.NewDecoder(r.Body).Decode(&itemReq); err != nil {
			// Handle decoding error
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		// Query database for item
		var item Item
		err = db.QueryRow("SELECT id, name, price FROM items WHERE id = '" + itemID + "'")
		if err != nil {
			// Handle database error
			http.Error(w, "Error retrieving item", http.StatusInternalServerError)
			return
		}

		// Encode item as JSON and respond
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(item); err != nil {
			// Handle encoding error
			http.Error(w, "Error encoding response", http.StatusInternalServerError)
			return
		}
	})

	// Start server
	fmt.Println("Server listening on port 8080")
	http.ListenAndServe(":8080", router)
}
