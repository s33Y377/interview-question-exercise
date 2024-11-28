Here are several MongoDB query exercises with solutions to help you learn and practice common operations in MongoDB:

### 1. **Find All Documents in a Collection**
   **Exercise:** Write a query to retrieve all documents from a collection called `users`.

   **Solution:**
   ```javascript
   db.users.find({})
   ```

### 2. **Find Documents with Specific Fields**
   **Exercise:** Retrieve all documents from the `users` collection where the `age` is greater than 25.

   **Solution:**
   ```javascript
   db.users.find({ age: { $gt: 25 } })
   ```

### 3. **Find Documents with Multiple Conditions**
   **Exercise:** Write a query to find all documents in the `users` collection where the `age` is greater than 25 and the `status` is "active".

   **Solution:**
   ```javascript
   db.users.find({ age: { $gt: 25 }, status: "active" })
   ```

### 4. **Find Documents with OR Condition**
   **Exercise:** Retrieve all users who are either older than 30 or have the status "inactive".

   **Solution:**
   ```javascript
   db.users.find({ $or: [{ age: { $gt: 30 } }, { status: "inactive" }] })
   ```

### 5. **Project Specific Fields**
   **Exercise:** Write a query to find all documents in the `users` collection but only return the `name` and `email` fields.

   **Solution:**
   ```javascript
   db.users.find({}, { name: 1, email: 1 })
   ```

### 6. **Limit the Number of Documents**
   **Exercise:** Retrieve only the first 5 documents from the `users` collection.

   **Solution:**
   ```javascript
   db.users.find().limit(5)
   ```

### 7. **Sort the Results**
   **Exercise:** Write a query to retrieve all documents from the `users` collection and sort them by `age` in ascending order.

   **Solution:**
   ```javascript
   db.users.find().sort({ age: 1 })
   ```

   **For descending order**:
   ```javascript
   db.users.find().sort({ age: -1 })
   ```

### 8. **Count the Number of Documents**
   **Exercise:** Count how many documents in the `users` collection have the `status` set to "active".

   **Solution:**
   ```javascript
   db.users.countDocuments({ status: "active" })
   ```

### 9. **Update a Document**
   **Exercise:** Write a query to update the `status` of the user with the `user_id` of 1001 to "inactive".

   **Solution:**
   ```javascript
   db.users.updateOne(
     { user_id: 1001 },
     { $set: { status: "inactive" } }
   )
   ```

### 10. **Update Multiple Documents**
   **Exercise:** Change the `status` to "active" for all users who have an `age` greater than 40.

   **Solution:**
   ```javascript
   db.users.updateMany(
     { age: { $gt: 40 } },
     { $set: { status: "active" } }
   )
   ```

### 11. **Delete a Document**
   **Exercise:** Write a query to delete a user with `user_id` 1001.

   **Solution:**
   ```javascript
   db.users.deleteOne({ user_id: 1001 })
   ```

### 12. **Delete Multiple Documents**
   **Exercise:** Delete all users whose `age` is less than 18.

   **Solution:**
   ```javascript
   db.users.deleteMany({ age: { $lt: 18 } })
   ```

### 13. **Find Documents with a Specific Pattern (Regular Expression)**
   **Exercise:** Find all users whose name starts with "A".

   **Solution:**
   ```javascript
   db.users.find({ name: { $regex: /^A/ } })
   ```

### 14. **Aggregation: Grouping Documents**
   **Exercise:** Write a query that calculates the average `age` of users in the `users` collection.

   **Solution:**
   ```javascript
   db.users.aggregate([
     { $group: { _id: null, averageAge: { $avg: "$age" } } }
   ])
   ```

### 15. **Aggregation: Grouping with Multiple Fields**
   **Exercise:** Group users by `status` and count how many users belong to each status.

   **Solution:**
   ```javascript
   db.users.aggregate([
     { $group: { _id: "$status", count: { $sum: 1 } } }
   ])
   ```

### 16. **Find Documents with a Specific Array Element**
   **Exercise:** Retrieve all users who have a `tags` array containing "developer".

   **Solution:**
   ```javascript
   db.users.find({ tags: "developer" })
   ```

### 17. **Find Documents with Array Length**
   **Exercise:** Find users whose `friends` array has more than 5 elements.

   **Solution:**
   ```javascript
   db.users.find({ friends: { $size: { $gt: 5 } } })
   ```

### 18. **Lookup (Join)**
   **Exercise:** Write a query to perform a join between the `users` and `orders` collections. You want to retrieve all users along with their orders.

   **Solution:**
   ```javascript
   db.users.aggregate([
     {
       $lookup: {
         from: "orders",
         localField: "user_id",
         foreignField: "user_id",
         as: "user_orders"
       }
     }
   ])
   ```

### 19. **Find Documents by ObjectId**
   **Exercise:** Write a query to find a document in the `users` collection by its `_id`.

   **Solution:**
   ```javascript
   db.users.find({ _id: ObjectId("5f6e9f0e8f6f20001fba9b1d") })
   ```

### 20. **Find Documents Using a GeoSpatial Query**
   **Exercise:** Find users who live within 10 kilometers of a specific geographic location (latitude: 40.7128, longitude: -74.0060).

   **Solution:**
   ```javascript
   db.users.createIndex({ location: "2dsphere" })
   db.users.find({
     location: {
       $near: {
         $geometry: {
           type: "Point",
           coordinates: [ -74.0060, 40.7128 ]
         },
         $maxDistance: 10000  // 10 kilometers
       }
     }
   })
   ```

### Conclusion:
These exercises cover basic CRUD operations, filtering, sorting, aggregation, and geospatial queries in MongoDB. Practicing these will give you a solid foundation in MongoDB queries and operations.
