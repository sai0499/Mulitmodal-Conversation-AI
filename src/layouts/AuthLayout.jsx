import React from 'react';
import './AuthLayout.css';

function AuthLayout({ children }) {
  return (
    <div className="auth-layout-container">
      {/* 
        This is the outer container for your Auth pages. 
        It provides the background and centering. 
      */}
      {children}
    </div>
  );
}

export default AuthLayout;
