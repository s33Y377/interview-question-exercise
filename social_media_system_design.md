Designing a database schema for an Instagram-like platform requires thoughtful planning, as it involves many features like user management, posts, comments, likes, followers, and interactions. Below is an outline of the database schema and the APIs needed for such a platform.

### 1. **Database Schema Design**

We’ll focus on essential tables, relationships, and relevant fields required for the platform.

#### **Tables**

1. **Users**
   - `user_id` (PK, int, auto_increment) — Unique identifier for each user.
   - `username` (varchar(255)) — Unique username.
   - `email` (varchar(255), unique) — User's email address.
   - `password` (varchar(255)) — Hashed password.
   - `profile_picture` (varchar(255)) — URL of the profile picture.
   - `bio` (text) — A short bio of the user.
   - `created_at` (timestamp) — Account creation date.
   - `updated_at` (timestamp) — Last updated date.
   
2. **Posts**
   - `post_id` (PK, int, auto_increment) — Unique post identifier.
   - `user_id` (FK to Users.user_id) — The user who created the post.
   - `image_url` (varchar(255)) — URL of the post image.
   - `caption` (text) — Caption for the post.
   - `created_at` (timestamp) — Date and time the post was created.
   - `updated_at` (timestamp) — Date and time the post was last updated.
   
3. **Comments**
   - `comment_id` (PK, int, auto_increment) — Unique comment identifier.
   - `post_id` (FK to Posts.post_id) — The post to which the comment belongs.
   - `user_id` (FK to Users.user_id) — The user who commented.
   - `comment` (text) — The comment text.
   - `created_at` (timestamp) — Date and time the comment was made.
   
4. **Likes**
   - `like_id` (PK, int, auto_increment) — Unique like identifier.
   - `post_id` (FK to Posts.post_id) — The post that was liked.
   - `user_id` (FK to Users.user_id) — The user who liked the post.
   - `created_at` (timestamp) — Date and time the like was made.

5. **Followers**
   - `follower_id` (PK, int, auto_increment) — Unique identifier for the follower relationship.
   - `user_id` (FK to Users.user_id) — The user who is being followed.
   - `follower_user_id` (FK to Users.user_id) — The user who follows.
   - `created_at` (timestamp) — Date and time the follow action occurred.

6. **Notifications**
   - `notification_id` (PK, int, auto_increment) — Unique identifier for notifications.
   - `user_id` (FK to Users.user_id) — The user who is receiving the notification.
   - `type` (varchar(50)) — Type of notification (like, comment, follow, etc.).
   - `reference_id` (int) — Reference to a specific post or user (like post_id or user_id).
   - `message` (text) — The notification content.
   - `read` (boolean) — Whether the notification has been read.
   - `created_at` (timestamp) — Date and time the notification was created.

7. **Tags**
   - `tag_id` (PK, int, auto_increment) — Unique identifier for tags.
   - `tag_name` (varchar(255)) — The tag text.
   
8. **Post_Tags**
   - `post_id` (FK to Posts.post_id) — The post associated with the tag.
   - `tag_id` (FK to Tags.tag_id) — The tag associated with the post.

---

### 2. **API Endpoints**

These API endpoints will allow users to interact with the platform.

#### **User Authentication**
- **POST /api/register** — Register a new user.
- **POST /api/login** — User login (returns JWT token).
- **POST /api/logout** — Log out the user (invalidate token).
- **POST /api/forgot-password** — Send reset password link.
- **POST /api/reset-password** — Reset the password using token.

#### **User Profile**
- **GET /api/user/{user_id}** — Get user profile details.
- **PUT /api/user/{user_id}** — Update user profile (profile picture, bio).
- **GET /api/user/{user_id}/followers** — Get a list of followers of the user.
- **GET /api/user/{user_id}/following** — Get a list of users the user is following.
- **POST /api/user/follow/{user_id}** — Follow a user.
- **POST /api/user/unfollow/{user_id}** — Unfollow a user.

#### **Posts**
- **POST /api/posts** — Create a new post (image + caption).
- **GET /api/posts/{post_id}** — Get details of a single post.
- **GET /api/posts** — Get a list of posts (with pagination).
- **PUT /api/posts/{post_id}** — Update a post (caption, image).
- **DELETE /api/posts/{post_id}** — Delete a post.

#### **Likes**
- **POST /api/posts/{post_id}/like** — Like a post.
- **POST /api/posts/{post_id}/unlike** — Unlike a post.
- **GET /api/posts/{post_id}/likes** — Get the list of users who liked a post.

#### **Comments**
- **POST /api/posts/{post_id}/comments** — Add a comment to a post.
- **GET /api/posts/{post_id}/comments** — Get all comments on a post.
- **DELETE /api/comments/{comment_id}** — Delete a comment.
  
#### **Notifications**
- **GET /api/notifications** — Get a list of notifications for a user.
- **PUT /api/notifications/{notification_id}/read** — Mark a notification as read.
- **DELETE /api/notifications/{notification_id}** — Delete a notification.

#### **Search**
- **GET /api/search/users?query={query}** — Search users by username.
- **GET /api/search/posts?query={query}** — Search posts by caption or tag.

#### **Tags**
- **GET /api/tags** — Get a list of all tags.
- **GET /api/tags/{tag_name}/posts** — Get all posts associated with a specific tag.

---

### 3. **API Design Considerations**

- **Authentication & Authorization**: Use JWT (JSON Web Token) for secure user authentication and authorization.
- **Rate Limiting**: Protect APIs like search, follow/unfollow, and like/unlike with rate limiting to prevent abuse.
- **Pagination**: Implement pagination on endpoints that return lists of data (e.g., posts, comments, followers).
- **Data Validation**: Use input validation on all fields, ensuring they meet the expected format (e.g., valid email format, image URL, etc.).
- **Caching**: Cache frequently accessed data like popular posts or trending tags.
- **Security**: Encrypt sensitive information, including passwords, using industry-standard algorithms like bcrypt.

---
---

This is a high-level design for the Instagram-like platform’s database schema and API structure. There may be additional features such as messaging, stories, or other social media-like interactions, depending on the specific requirements of the platform.

---

### Get like count for a post

When dealing with a post that has millions of likes, retrieving the like count efficiently becomes important for performance. To ensure that you can retrieve the like count in a performant way without directly counting every like every time, here are a few strategies you can use:

### 1. **Use a Cached Count (Optimized Storage)**
One of the most efficient ways to handle this is to maintain a **like count** in a separate table or as an attribute within the `Posts` table itself. Instead of querying the entire `Likes` table every time you need to know the like count, you update this cached value only when a user likes or unlikes the post.

#### **Table Design with Cached Count**

You can add a `like_count` column to the `Posts` table to store the number of likes:

```sql
ALTER TABLE Posts
ADD COLUMN like_count INT DEFAULT 0;
```

This column will store the current like count for each post.

#### **API Logic**
- **Incrementing the like count**: When a user likes a post, increment the `like_count` field in the `Posts` table.
- **Decrementing the like count**: When a user unlikes a post, decrement the `like_count` field.

#### **SQL Queries for Incrementing and Decrementing**
1. **Like a post** (Increment the like count):
   ```sql
   UPDATE Posts
   SET like_count = like_count + 1
   WHERE post_id = {post_id};
   ```

2. **Unlike a post** (Decrement the like count):
   ```sql
   UPDATE Posts
   SET like_count = like_count - 1
   WHERE post_id = {post_id};
   ```

3. **Get the like count for a post**:
   ```sql
   SELECT like_count
   FROM Posts
   WHERE post_id = {post_id};
   ```

### 2. **Using a Counter Service with Eventual Consistency**

In systems where the like count needs to be highly scalable (e.g., millions of likes), another option is to use a **distributed counter** or **eventual consistency** approach. This can be achieved using caching systems like Redis, where you can use atomic operations to increment and retrieve the like count.

#### **Redis Approach**

You can use Redis to store the like count and increment it atomically. Redis provides an efficient way to handle counters without locking or affecting database performance.

1. **Increment the Like Count**:
   ```python
   import redis

   r = redis.StrictRedis(host='localhost', port=6379, db=0)
   post_id = "post:12345"  # Example post ID
   r.incr(post_id)  # Increments the like count by 1
   ```

2. **Get the Like Count**:
   ```python
   like_count = r.get(post_id)  # Retrieves the like count
   if like_count is None:
       like_count = 0  # If the post has no likes, initialize to 0
   ```

3. **Ensure Eventual Consistency**:  
   In some cases, you might want to periodically synchronize the Redis counter with your primary database (e.g., at regular intervals or during off-peak hours). This ensures that the count in Redis doesn’t drift too far from the actual database value.

   **Synchronizing with Database**:
   - When the like count reaches a threshold or after a set period, batch update the main `Posts` table with the like count stored in Redis.

### 3. **Sharding for Even Larger Scales**

If the number of likes for a post is exceptionally large (millions), you can consider sharding or partitioning the `Likes` table across multiple database nodes. This approach ensures that querying likes doesn't overload any single database server.

For example:
- **Sharding**: Split the `Likes` table into multiple partitions based on certain criteria, such as ranges of post IDs or geographical locations. This reduces the load on a single server when querying like counts.

### 4. **Pre-aggregated Likes in a Separate Table**

Another approach to prevent excessive querying of millions of likes is to maintain an **aggregated like count** in a separate table that updates periodically (e.g., using a cron job or a background process).

#### **Likes Aggregate Table**

You can create a table that stores the aggregated like counts for posts. The application can then query this table for the like count rather than the `Likes` table itself.

```sql
CREATE TABLE Post_Like_Aggregates (
    post_id INT PRIMARY KEY,
    like_count INT DEFAULT 0
);
```

This table can be updated through an automated process:
- **Incremental Updates**: When a new like is added, update the `Post_Like_Aggregates` table in a batch process (perhaps every few minutes).
- **Real-Time Updates**: If a post receives an exceptionally high number of likes, you can update the like count in real-time, using a background process.

---

### 5. **API Endpoint to Get Like Count for a Post**
Once you have a mechanism to efficiently store and retrieve the like count, the corresponding API endpoint to fetch the like count would look like this:

#### **GET /api/posts/{post_id}/like_count**

- **Description**: Fetch the like count for a post.
- **Response Example**:
  ```json
  {
    "post_id": 12345,
    "like_count": 5000000
  }
  ```

- **Implementation**:  
  Depending on your system design (Redis, cached database, etc.), the server would return the like count in response to the user request.

---

### Summary
- For a post with millions of likes, it's crucial to **cache the like count** (either in the database or using a distributed system like Redis).
- You can **increment and decrement** the like count in the `Posts` table when a user likes or unlikes a post.
- For extremely large numbers of likes, you may need to use **sharding**, **eventual consistency**, or a **background process** to aggregate and sync the like counts.


---

To implement an API endpoint for fetching posts for a feed on scroll, along with their like and comment counts, you will need a combination of pagination, efficient counting for likes and comments, and optimizing data retrieval to handle large datasets (especially with millions of posts, likes, and comments).

Below, I’ll outline how to design the system to fetch posts efficiently, including like and comment counts, when a user scrolls through the feed.

### Key Features of the Feed:
1. **Pagination**: Use pagination (often in combination with offset/limit or cursor-based pagination) to fetch a subset of posts at a time.
2. **Efficient Like and Comment Count**: Calculate the like and comment counts efficiently, ideally without querying every like and comment each time.
3. **Caching**: To prevent unnecessary database load, you can cache the like and comment counts.

### Database Schema Assumptions

You have the following tables:

1. **Posts**: Stores the posts.
2. **Likes**: Stores likes for posts.
3. **Comments**: Stores comments for posts.

Here’s a quick summary of these tables:
- `Posts`: Contains the posts (`post_id`, `user_id`, `caption`, `image_url`, `created_at`).
- `Likes`: Tracks likes for posts (`post_id`, `user_id`, `created_at`).
- `Comments`: Stores comments for posts (`comment_id`, `post_id`, `user_id`, `comment_text`, `created_at`).

We will also assume you have a `like_count` and `comment_count` in the `Posts` table or some caching mechanism to avoid recalculating these counts on every scroll.

### 1. **Caching Like and Comment Counts in the `Posts` Table (Optional but Recommended)**
To improve performance, maintain a **cached count** for both likes and comments in the `Posts` table. This helps reduce the number of queries to the `Likes` and `Comments` tables.

```sql
ALTER TABLE Posts
ADD COLUMN like_count INT DEFAULT 0,
ADD COLUMN comment_count INT DEFAULT 0;
```

Whenever a like or comment is added or removed, you update these counts in the `Posts` table. This approach reduces the need for querying large datasets to count likes and comments on every scroll.

### 2. **Pagination for the Feed**
For efficient pagination, we use **cursor-based pagination**. This ensures that each scroll fetches the next set of posts based on a unique identifier (e.g., `post_id` or `created_at`) and avoids issues with offsets when data changes.

#### Cursor-based Pagination Query:
Let's say you're showing posts sorted by their `created_at` or `post_id`, and you fetch posts starting from a certain point.

```sql
SELECT post_id, user_id, caption, image_url, created_at, like_count, comment_count
FROM Posts
WHERE created_at <= {last_seen_post_created_at}
ORDER BY created_at DESC
LIMIT 20;
```

This query fetches the next 20 posts (or whatever the desired page size is) based on the `created_at` timestamp or `post_id`.

- **{last_seen_post_created_at}**: This is the timestamp of the last post from the previous page (for pagination).
- **ORDER BY created_at DESC**: Ensures posts are displayed in descending order by their creation date.

If you prefer using `post_id` instead of `created_at` for sorting:

```sql
SELECT post_id, user_id, caption, image_url, created_at, like_count, comment_count
FROM Posts
WHERE post_id < {last_seen_post_id}
ORDER BY post_id DESC
LIMIT 20;
```

### 3. **API Endpoint Design**

#### **GET /api/feed**
- **Description**: Fetch posts for the user's feed on scroll, along with the like and comment counts.
- **Request Parameters**:
  - `last_seen_post_id`: The `post_id` of the last post the user saw (for pagination).
  - `limit`: The number of posts to return per request (default can be 20).
- **Response**:
  - A list of posts with their `like_count` and `comment_count`.

#### API Example:

##### **Request**:
```http
GET /api/feed?last_seen_post_id=123456&limit=20
```

##### **Response** (JSON):
```json
{
  "posts": [
    {
      "post_id": 123457,
      "user_id": 10,
      "caption": "Beautiful sunset!",
      "image_url": "https://example.com/images/post1.jpg",
      "created_at": "2024-12-20T10:15:30Z",
      "like_count": 1200,
      "comment_count": 300
    },
    {
      "post_id": 123458,
      "user_id": 15,
      "caption": "Had a great time at the beach!",
      "image_url": "https://example.com/images/post2.jpg",
      "created_at": "2024-12-20T09:50:00Z",
      "like_count": 2500,
      "comment_count": 450
    },
    ...
  ]
}
```

### 4. **Handling Like and Comment Counts**

#### **Efficient Count Retrieval for Posts**:
To handle the like and comment counts without querying the entire `Likes` and `Comments` tables each time, use the following strategies:

- **Cached Count**: If you're storing the like and comment counts in the `Posts` table (as mentioned earlier), simply fetch these values directly from the `Posts` table in the pagination query.
- **Incrementing/Decrementing Counts**: Whenever a user likes or unlikes a post, or adds/removes a comment, update the respective `like_count` and `comment_count` fields in the `Posts` table.
  
#### **Updating Counts on Like/Comment Action**:
For example:
- When a user likes a post:
  ```sql
  INSERT INTO Likes (post_id, user_id, created_at) VALUES ({post_id}, {user_id}, NOW());
  UPDATE Posts SET like_count = like_count + 1 WHERE post_id = {post_id};
  ```

- When a user unlikes a post:
  ```sql
  DELETE FROM Likes WHERE post_id = {post_id} AND user_id = {user_id};
  UPDATE Posts SET like_count = like_count - 1 WHERE post_id = {post_id};
  ```

- When a user adds a comment:
  ```sql
  INSERT INTO Comments (post_id, user_id, comment_text, created_at) VALUES ({post_id}, {user_id}, {comment_text}, NOW());
  UPDATE Posts SET comment_count = comment_count + 1 WHERE post_id = {post_id};
  ```

### 5. **Rate Limiting and Caching**

To avoid excessive database hits for counting likes and comments, implement **caching** for frequent queries. You can use a caching layer like Redis for the like and comment counts, and invalidate/recalculate these values periodically or when a like/comment is added/removed.

---

### Summary

- **Pagination**: Use cursor-based pagination to efficiently load posts for the feed.
- **Like and Comment Counts**: Maintain counts in the `Posts` table for quick access.
- **API Endpoint**: Implement a `GET /api/feed` endpoint that returns a list of posts with the like and comment counts.
- **Efficient Counting**: Increment/decrement the like and comment counts in the `Posts` table when a like or comment is added or removed.
- **Caching**: Consider caching like and comment counts using Redis for scalability.

This approach should help in efficiently delivering posts with their like and comment counts as users scroll through the feed.

---
---

Direct messaging (DM) in a social media application like Instagram allows users to send and receive text and media messages (images, videos, etc.) privately. Designing a system for DMs requires considering the following aspects:

1. **Database Schema for DMs**: 
   - Storing messages.
   - Storing media (images/videos).
   - Efficiently fetching messages.

2. **APIs for Sending/Receiving Text and Media Messages**: 
   - Text message sending.
   - Media message sending.
   - Message retrieval.

3. **Handling Notifications**: 
   - Notifying users when they receive a new message.

### 1. **Database Schema Design for Direct Messages**

To handle Direct Messages (DMs) effectively, we need to design a schema that can store both text and media messages. This schema will also support efficient retrieval of messages, as well as maintain the relationships between users.

#### Tables:

1. **Users**:
   This table stores user data (`user_id`, `username`, `email`, etc.).

2. **Conversations**:
   This table represents a conversation between two or more users.
   - `conversation_id` (PK) — Unique ID for the conversation.
   - `created_at` (timestamp) — The time when the conversation started.
   - `updated_at` (timestamp) — The time when the conversation was last updated.
   - `is_active` (boolean) — Whether the conversation is still active (if all messages are deleted, set to inactive).

3. **Conversation_Participants**:
   This table stores the participants in a conversation.
   - `conversation_id` (FK) — ID of the conversation.
   - `user_id` (FK) — ID of the user participating in the conversation.
   - `is_read` (boolean) — Whether the user has read the latest message.

4. **Messages**:
   This table stores the messages within a conversation (both text and media).
   - `message_id` (PK) — Unique ID for each message.
   - `conversation_id` (FK) — The conversation to which the message belongs.
   - `sender_id` (FK to Users.user_id) — The user who sent the message.
   - `message_type` (enum: `text`, `image`, `video`, etc.) — Type of message.
   - `message_content` (text) — The actual content of the message (text, URL of media file, etc.).
   - `created_at` (timestamp) — Timestamp of when the message was sent.
   - `is_read` (boolean) — Whether the recipient has read the message.

5. **Media**:
   This table stores media files (if a message contains an image or video).
   - `media_id` (PK) — Unique media ID.
   - `message_id` (FK to Messages.message_id) — The message associated with the media.
   - `media_type` (enum: `image`, `video`) — Type of media.
   - `file_url` (varchar) — URL pointing to the media file (e.g., stored in cloud storage like AWS S3).
   - `created_at` (timestamp) — Timestamp when the media was uploaded.

#### Example Schema:
```sql
CREATE TABLE Conversations (
    conversation_id INT AUTO_INCREMENT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE Conversation_Participants (
    conversation_id INT,
    user_id INT,
    is_read BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (conversation_id) REFERENCES Conversations(conversation_id),
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);

CREATE TABLE Messages (
    message_id INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    sender_id INT,
    message_type ENUM('text', 'image', 'video'),
    message_content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_read BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (conversation_id) REFERENCES Conversations(conversation_id),
    FOREIGN KEY (sender_id) REFERENCES Users(user_id)
);

CREATE TABLE Media (
    media_id INT AUTO_INCREMENT PRIMARY KEY,
    message_id INT,
    media_type ENUM('image', 'video'),
    file_url VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (message_id) REFERENCES Messages(message_id)
);
```

### 2. **API Endpoints for Sending/Receiving Text and Media Messages**

#### **Send Text Message**

- **POST /api/messages/send_text**
  - **Request**:
    ```json
    {
      "conversation_id": 123,
      "sender_id": 1,
      "message_content": "Hello, how are you?",
      "message_type": "text"
    }
    ```
  - **Response**:
    ```json
    {
      "message_id": 456,
      "conversation_id": 123,
      "sender_id": 1,
      "message_content": "Hello, how are you?",
      "message_type": "text",
      "created_at": "2024-12-20T10:15:30Z",
      "is_read": false
    }
    ```

  **API Logic**:
  - Create a new `Message` record in the `Messages` table with `message_type = 'text'`.
  - Update the `updated_at` field in the `Conversations` table.
  - Notify the recipient (either via push notification or internal system).

#### **Send Media Message (Image/Video)**

- **POST /api/messages/send_media**
  - **Request**:
    ```json
    {
      "conversation_id": 123,
      "sender_id": 1,
      "message_type": "image",
      "media_url": "https://example.com/media/image.jpg"
    }
    ```
  - **Response**:
    ```json
    {
      "message_id": 457,
      "conversation_id": 123,
      "sender_id": 1,
      "message_content": "https://example.com/media/image.jpg",
      "message_type": "image",
      "created_at": "2024-12-20T10:20:00Z",
      "is_read": false
    }
    ```

  **API Logic**:
  - First, store the media file (image or video) on cloud storage (e.g., AWS S3).
  - Create a new `Message` record with the `message_type = 'image'` (or `video`).
  - Create an associated `Media` record linking to the media URL and the corresponding message.
  - Update the conversation's `updated_at` field and notify the recipient.

#### **Get Messages for a Conversation**

- **GET /api/messages/{conversation_id}**
  - **Request**:
    ```http
    GET /api/messages/123
    ```
  - **Response**:
    ```json
    {
      "conversation_id": 123,
      "messages": [
        {
          "message_id": 456,
          "sender_id": 1,
          "message_type": "text",
          "message_content": "Hello, how are you?",
          "created_at": "2024-12-20T10:15:30Z",
          "is_read": false
        },
        {
          "message_id": 457,
          "sender_id": 2,
          "message_type": "image",
          "message_content": "https://example.com/media/image.jpg",
          "created_at": "2024-12-20T10:20:00Z",
          "is_read": false
        }
      ]
    }
    ```

  **API Logic**:
  - Fetch all messages for the given `conversation_id` from the `Messages` table.
  - Fetch associated media from the `Media` table (if any).

#### **Mark Message as Read**

- **POST /api/messages/mark_as_read**
  - **Request**:
    ```json
    {
      "message_id": 456,
      "user_id": 1
    }
    ```
  - **Response**:
    ```json
    {
      "message_id": 456,
      "is_read": true
    }
    ```

  **API Logic**:
  - Mark the `is_read` field as `true` for the specified `message_id` in the `Messages` table.

---

### 3. **Notification System**

You can implement notifications to alert users when they receive a new message. The notification can be sent when a new message is inserted into the `Messages` table or when the recipient has not read the message.

- **Push Notification**: Use Firebase Cloud Messaging (FCM) or a similar service to send push notifications to users when they receive a new message.
- **In-app Notifications**: Display notifications within the app interface.

---

### 4. **Example Flow for Sending a Media Message**

1. **User A sends a text message**: 
   - User A sends a text message to User B via the `/api/messages/send_text` endpoint.
   - The system creates the message in the `Messages` table and notifies User B.

2. **User A sends a media message**:
   - User A uploads an image through the app, which is then stored in a cloud storage service.
   - The app calls the `/api/messages/send_media` endpoint with the URL of the image.
   - The system stores the media URL in the `Media` table and creates a corresponding message in the `Messages` table.

3. **User B reads the message**:
   - User B fetches the messages through the `/api/messages/{conversation_id}` endpoint.
   - User B marks the message as read, and the system updates the `is_read` field.

---

### Summary

1. **Database Design**: Use tables for `Conversations`, `Conversation_Participants`, `Messages`, and `Media` to store text and media messages.
2. **API Endpoints**: Provide endpoints for sending text messages, media messages (images/videos), and retrieving messages.
3. **Notifications**: Use push or in-app notifications to alert users about new messages.
4. **Media Storage**: Store media files (images/videos) in cloud storage (e.g., AWS S3) and link them to messages via URLs.

This design will allow you to handle both text and media messages efficiently while providing a seamless experience for users in your messaging system.

---

