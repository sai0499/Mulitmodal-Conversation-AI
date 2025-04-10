import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

// Your components
import AuthForm from './components/AuthForm';
import ConversationPage from './components/ConversationPage';

// (Optional) If you still want a layout for auth:
import AuthLayout from './layouts/AuthLayout';

// Global app CSS
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        {/* Login (uses AuthLayout if you still want that gradient) */}
        <Route
          path="/"
          element={
            <AuthLayout>
              <AuthForm />
            </AuthLayout>
          }
        />

        {/* Conversation (directly uses its own CSS, not ConversationLayout) */}
        <Route
          path="/conversation"
          element={<ConversationPage />}
        />
      </Routes>
    </Router>
  );
}

export default App;
