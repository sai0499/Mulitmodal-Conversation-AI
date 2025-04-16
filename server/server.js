require('dotenv').config();
const express = require('express');
const cors = require('cors');
const connectDB = require('./config/db');
const authRoutes = require('./routes/authRoutes');
const conversationRoutes = require('./routes/conversationRoutes');
const userRoutes = require('./routes/userRoutes'); // New user routes

const app = express();

// Body parsing and CORS
app.use(express.json());
app.use(cors());

// Connect to MongoDB
connectDB(process.env.MONGO_URI);

// Setup routes
app.use('/api/auth', authRoutes);
app.use('/api', conversationRoutes);  // Conversation endpoints
app.use('/api/user', userRoutes);       // User API key endpoints

// Test route
app.get('/', (req, res) => {
  res.send('AI Conversation Auth Server is running...');
});

// Start server
const PORT = process.env.PORT || 4000;
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server listening on port ${PORT}`);
});
