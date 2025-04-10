require('dotenv').config();
const express = require('express');
const cors = require('cors');
const connectDB = require('./config/db');
const authRoutes = require('./routes/authRoutes');
const conversationRoutes = require('./routes/conversationRoutes');

const app = express();

// Body parsing and CORS
app.use(express.json());
app.use(cors());

// Connect to MongoDB
connectDB(process.env.MONGO_URI);

// Setup routes
app.use('/api/auth', authRoutes);
app.use('/api', conversationRoutes);  // Conversations endpoint(s)

// Test route
app.get('/', (req, res) => {
  res.send('AI Conversation Auth Server is running...');
});

// Start server
const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
