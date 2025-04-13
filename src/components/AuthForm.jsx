import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom'; // <-- Import from React Router
import './AuthForm.css';

/**
 * AuthForm component handles:
 *  1) Login (matricNumber + password)
 *  2) Sign-up (OTP flow)
 *  3) Forgot Password (OTP flow)
 *
 * Replaces 'alert()' with an in-page success message.
 */
function AuthForm() {
  // 'login' | 'signup' | 'forgot'
  const [mode, setMode] = useState('login');

  // ---------------------- LOGIN STATES ----------------------
  const [loginMatric, setLoginMatric] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [loginShowPassword, setLoginShowPassword] = useState(false);

  // ---------------------- SIGN-UP STATES ----------------------
  const [signupMatric, setSignupMatric] = useState('');
  const [signupEmail, setSignupEmail] = useState('');
  const [signupOTP, setSignupOTP] = useState('');
  const [signupPassword, setSignupPassword] = useState('');
  const [signUpConfirmPassword, setSignUpConfirmPassword] = useState('');
  const [isSignUpOTPSent, setIsSignUpOTPSent] = useState(false);
  const [isSignUpOTPVerified, setIsSignUpOTPVerified] = useState(false);
  const [signUpShowPassword, setSignUpShowPassword] = useState(false);

  // ---------------------- FORGOT PASSWORD STATES ----------------------
  const [forgotMatric, setForgotMatric] = useState('');
  const [forgotEmail, setForgotEmail] = useState('');
  const [forgotOTP, setForgotOTP] = useState('');
  const [forgotNewPassword, setForgotNewPassword] = useState('');
  const [forgotConfirmPassword, setForgotConfirmPassword] = useState('');
  const [isForgotOTPSent, setIsForgotOTPSent] = useState(false);
  const [isForgotOTPVerified, setIsForgotOTPVerified] = useState(false);
  const [forgotShowPassword, setForgotShowPassword] = useState(false);

  // ---------------------- MESSAGES ----------------------
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  // React Router navigate hook
  const navigate = useNavigate();

  /** Reset all states before switching mode. */
  const resetAllStates = () => {
    // Login
    setLoginMatric('');
    setLoginPassword('');
    setLoginShowPassword(false);
    // Sign Up
    setSignupMatric('');
    setSignupEmail('');
    setSignupOTP('');
    setSignupPassword('');
    setSignUpConfirmPassword('');
    setIsSignUpOTPSent(false);
    setIsSignUpOTPVerified(false);
    setSignUpShowPassword(false);
    // Forgot
    setForgotMatric('');
    setForgotEmail('');
    setForgotOTP('');
    setForgotNewPassword('');
    setForgotConfirmPassword('');
    setIsForgotOTPSent(false);
    setIsForgotOTPVerified(false);
    setForgotShowPassword(false);
    // Messages
    setErrorMessage('');
    setSuccessMessage('');
  };

  /** Switch mode: "login", "signup", or "forgot". */
  const handleSwitchMode = (newMode) => {
    resetAllStates();
    setMode(newMode);
  };

  /* ------------------------------------------------------------------
   *                     LOGIN FLOW
   * ------------------------------------------------------------------*/
  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    try {
      if (!loginMatric || !loginPassword) {
        setSuccessMessage('');
        setErrorMessage('Please enter both Matriculation Number and Password.');
        return;
      }
      setErrorMessage('');
      setSuccessMessage('');

      const response = await fetch('http://localhost:4000/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          matricNumber: loginMatric,
          password: loginPassword,
        }),
      });
      const data = await response.json();

      if (!data.success) {
        setErrorMessage(data.message || 'Login failed. Please try again.');
        setSuccessMessage('');
        return;
      }

      // SUCCESS: Save token and matric number in sessionstorage
      sessionStorage.setItem('token', data.token);
      sessionStorage.setItem('matricNumber', data.user.matricNumber);

      setErrorMessage('');
      setSuccessMessage('Login successful!');

      // Redirect to conversation page
      navigate('/conversation');
    } catch (err) {
      console.error(err);
      setErrorMessage('Error during login. Please try again.');
      setSuccessMessage('');
    }
  };

  /* ------------------------------------------------------------------
   *                     SIGN-UP FLOW
   * ------------------------------------------------------------------*/
  /** 1. SEND OTP for Sign Up */
  const handleSignUpSendOTP = async () => {
    try {
      if (!signupMatric) {
        setErrorMessage('Please enter your Matriculation Number first.');
        setSuccessMessage('');
        return;
      }
      setErrorMessage('');
      setSuccessMessage('');

      const response = await fetch('http://localhost:4000/api/auth/send-otp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          matricNumber: signupMatric,
          forSignup: true,
        }),
      });
      const data = await response.json();

      if (!data.success) {
        setErrorMessage(data.message || 'Error sending OTP.');
        setSuccessMessage('');
        return;
      }

      setSignupEmail(data.email || '');
      setIsSignUpOTPSent(true);
      setErrorMessage('');
      setSuccessMessage('Sign-Up OTP has been sent to your email address!');
    } catch (err) {
      console.error(err);
      setErrorMessage('Error sending OTP for Sign Up.');
      setSuccessMessage('');
    }
  };

  /** 2. VERIFY OTP for Sign Up */
  const handleSignUpVerifyOTP = async () => {
    try {
      if (!signupOTP) {
        setErrorMessage('Please enter the OTP you received.');
        setSuccessMessage('');
        return;
      }
      setErrorMessage('');
      setSuccessMessage('');

      const response = await fetch('http://localhost:4000/api/auth/verify-otp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          matricNumber: signupMatric,
          otp: signupOTP,
        }),
      });
      const data = await response.json();

      if (!data.success) {
        setErrorMessage(data.message || 'Incorrect or expired OTP.');
        setSuccessMessage('');
        return;
      }

      setIsSignUpOTPVerified(true);
      setErrorMessage('');
      setSuccessMessage('Sign Up OTP verified successfully!');
    } catch (err) {
      console.error(err);
      setErrorMessage('Error verifying OTP for Sign Up.');
      setSuccessMessage('');
    }
  };

  /** 3. FINALIZE SIGNUP */
  const handleSignUpSubmit = async (e) => {
    e.preventDefault();
    try {
      if (!signupMatric || !signupPassword || !signUpConfirmPassword) {
        setErrorMessage('Please fill all required fields.');
        setSuccessMessage('');
        return;
      }
      if (!isSignUpOTPVerified) {
        setErrorMessage('Please verify your OTP first.');
        setSuccessMessage('');
        return;
      }
      if (signupPassword !== signUpConfirmPassword) {
        setErrorMessage('Passwords do not match.');
        setSuccessMessage('');
        return;
      }

      setErrorMessage('');
      setSuccessMessage('');

      const response = await fetch('http://localhost:4000/api/auth/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          matricNumber: signupMatric,
          email: signupEmail,
          password: signupPassword,
        }),
      });
      const data = await response.json();

      if (!data.success) {
        setErrorMessage(data.message || 'Sign-up failed. Please try again.');
        setSuccessMessage('');
        return;
      }

      setErrorMessage('');
      setSuccessMessage('Sign-up successful!');

      // Switch to login mode after sign-up
      // (Optional short timeout or immediate switch)
      handleSwitchMode('login');
    } catch (err) {
      console.error(err);
      setErrorMessage('Error during sign up. Please try again.');
      setSuccessMessage('');
    }
  };

  /* ------------------------------------------------------------------
   *                     FORGOT PASSWORD FLOW
   * ------------------------------------------------------------------*/
  /** 1. SEND OTP for Forgot Password */
  const handleForgotSendOTP = async () => {
    try {
      if (!forgotMatric) {
        setErrorMessage('Please enter your Matriculation Number.');
        setSuccessMessage('');
        return;
      }
      setErrorMessage('');
      setSuccessMessage('');

      const response = await fetch('http://localhost:4000/api/auth/send-otp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          matricNumber: forgotMatric,
          forSignup: false, // for password reset
        }),
      });
      const data = await response.json();

      if (!data.success) {
        setErrorMessage(data.message || 'Error sending reset OTP.');
        setSuccessMessage('');
        return;
      }

      setForgotEmail(data.email || '');
      setIsForgotOTPSent(true);
      setErrorMessage('');
      setSuccessMessage('A reset OTP has been sent to your email!');
    } catch (err) {
      console.error(err);
      setErrorMessage('Error sending OTP for Password Reset.');
      setSuccessMessage('');
    }
  };

  /** 2. VERIFY OTP for Forgot Password */
  const handleForgotVerifyOTP = async () => {
    try {
      if (!forgotOTP) {
        setErrorMessage('Please enter the OTP.');
        setSuccessMessage('');
        return;
      }
      setErrorMessage('');
      setSuccessMessage('');

      const response = await fetch('http://localhost:4000/api/auth/verify-otp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          matricNumber: forgotMatric,
          otp: forgotOTP,
        }),
      });
      const data = await response.json();

      if (!data.success) {
        setErrorMessage(data.message || 'Incorrect or expired OTP.');
        setSuccessMessage('');
        return;
      }

      setIsForgotOTPVerified(true);
      setErrorMessage('');
      setSuccessMessage('Password Reset OTP verified successfully!');
    } catch (err) {
      console.error(err);
      setErrorMessage('Error verifying OTP for Password Reset.');
      setSuccessMessage('');
    }
  };

  /** 3. RESET PASSWORD */
  const handleForgotSubmit = async (e) => {
    e.preventDefault();
    try {
      if (!forgotMatric || !forgotNewPassword || !forgotConfirmPassword) {
        setErrorMessage('Please fill all required fields.');
        setSuccessMessage('');
        return;
      }
      if (!isForgotOTPVerified) {
        setErrorMessage('Please verify your OTP first.');
        setSuccessMessage('');
        return;
      }
      if (forgotNewPassword !== forgotConfirmPassword) {
        setErrorMessage('Passwords do not match.');
        setSuccessMessage('');
        return;
      }

      setErrorMessage('');
      setSuccessMessage('');

      const response = await fetch('http://localhost:4000/api/auth/reset-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          matricNumber: forgotMatric,
          newPassword: forgotNewPassword,
        }),
      });
      const data = await response.json();

      if (!data.success) {
        setErrorMessage(data.message || 'Password reset failed.');
        setSuccessMessage('');
        return;
      }

      setErrorMessage('');
      setSuccessMessage('Password reset successful!');
      // Switch to login mode after resetting password
      handleSwitchMode('login');
    } catch (err) {
      console.error(err);
      setErrorMessage('Error resetting password. Please try again.');
      setSuccessMessage('');
    }
  };

  return (
    <div className="auth-form-wrapper">
      <div className="auth-form-header">
        <div className="logo-title">
          <img src="../favicon.svg" alt="Logo" className="logo" />
          <span className="title">Uni-Ask</span>
        </div>
        {mode === 'login' && <h2>Sign In</h2>}
        {mode === 'signup' && <h2>Create your account</h2>}
        {mode === 'forgot' && <h2>Reset your password</h2>}
      </div>

      {/* SUCCESS and ERROR MESSAGES */}
      {successMessage && <div className="auth-form-success">{successMessage}</div>}
      {errorMessage && <div className="auth-form-error">{errorMessage}</div>}

      {/* ------------------- LOGIN FORM ------------------- */}
      {mode === 'login' && (
        <form onSubmit={handleLoginSubmit} className="auth-form">
          <label htmlFor="loginMatric">Matriculation Number</label>
          <input
            id="loginMatric"
            type="text"
            placeholder="Matric Number"
            value={loginMatric}
            onChange={(e) => setLoginMatric(e.target.value)}
          />

          <label htmlFor="loginPassword">Password</label>
          <input
            id="loginPassword"
            type={loginShowPassword ? 'text' : 'password'}
            placeholder="Password"
            value={loginPassword}
            onChange={(e) => setLoginPassword(e.target.value)}
          />

          {/* Show Password Checkbox */}
          <div className="checkbox-container">
            <input
              type="checkbox"
              id="loginShowPassword"
              checked={loginShowPassword}
              onChange={(e) => setLoginShowPassword(e.target.checked)}
            />
            <label htmlFor="loginShowPassword">Show Password</label>
          </div>

          <button type="submit" className="auth-form-btn">
            Continue
          </button>

          <div className="auth-form-footer">
            <p className="small-text link" onClick={() => handleSwitchMode('forgot')}>
              Forgot password?
            </p>
            <p>
              Don’t have an account?{' '}
              <span className="link" onClick={() => handleSwitchMode('signup')}>
                Sign Up
              </span>
            </p>
          </div>
        </form>
      )}

      {/* ------------------- SIGN-UP FORM ------------------- */}
      {mode === 'signup' && (
        <form onSubmit={handleSignUpSubmit} className="auth-form">
          <label htmlFor="signupMatric">Matriculation Number</label>
          <input
            id="signupMatric"
            type="text"
            placeholder="Matric Number"
            value={signupMatric}
            onChange={(e) => setSignupMatric(e.target.value)}
            disabled={isSignUpOTPVerified}
          />

          <label htmlFor="signupEmail">Email</label>
          <input
            id="signupEmail"
            type="email"
            placeholder="Email appears after OTP is sent"
            value={signupEmail}
            onChange={(e) => setSignupEmail(e.target.value)}
            disabled
          />

          {!isSignUpOTPSent && (
            <button
              type="button"
              className="auth-form-btn secondary"
              onClick={handleSignUpSendOTP}
            >
              Send OTP
            </button>
          )}

          {isSignUpOTPSent && !isSignUpOTPVerified && (
            <>
              <label htmlFor="signupOTP">Enter OTP</label>
              <input
                id="signupOTP"
                type="text"
                placeholder="6-digit OTP"
                value={signupOTP}
                onChange={(e) => setSignupOTP(e.target.value)}
              />
              <button
                type="button"
                className="auth-form-btn secondary"
                onClick={handleSignUpVerifyOTP}
              >
                Verify OTP
              </button>
            </>
          )}

          <label htmlFor="signupPassword">New Password</label>
          <input
            id="signupPassword"
            type={signUpShowPassword ? 'text' : 'password'}
            placeholder="Choose a password"
            value={signupPassword}
            onChange={(e) => setSignupPassword(e.target.value)}
            disabled={!isSignUpOTPVerified}
          />

          <label htmlFor="signUpConfirmPassword">Confirm Password</label>
          <input
            id="signUpConfirmPassword"
            type={signUpShowPassword ? 'text' : 'password'}
            placeholder="Re-enter your password"
            value={signUpConfirmPassword}
            onChange={(e) => setSignUpConfirmPassword(e.target.value)}
            disabled={!isSignUpOTPVerified}
          />

          {/* Show Password Checkbox */}
          {isSignUpOTPSent && (
            <div className="checkbox-container">
              <input
                type="checkbox"
                id="signUpShowPassword"
                checked={signUpShowPassword}
                onChange={(e) => setSignUpShowPassword(e.target.checked)}
                disabled={!isSignUpOTPSent}
              />
              <label htmlFor="signUpShowPassword">Show Password</label>
            </div>
          )}

          <button
            type="submit"
            className="auth-form-btn"
            disabled={!isSignUpOTPVerified}
          >
            Create Account
          </button>

          <div className="auth-form-footer">
            <p>
              Already have an account?{' '}
              <span className="link" onClick={() => handleSwitchMode('login')}>
                Sign in
              </span>
            </p>
          </div>
        </form>
      )}

      {/* ------------------- FORGOT PASSWORD FORM ------------------- */}
      {mode === 'forgot' && (
        <form onSubmit={handleForgotSubmit} className="auth-form">
          <label htmlFor="forgotMatric">Matriculation Number</label>
          <input
            id="forgotMatric"
            type="text"
            placeholder="Matric Number"
            value={forgotMatric}
            onChange={(e) => setForgotMatric(e.target.value)}
            disabled={isForgotOTPVerified}
          />

          <label htmlFor="forgotEmail">Email</label>
          <input
            id="forgotEmail"
            type="email"
            placeholder="Email appears after OTP is sent"
            value={forgotEmail}
            onChange={(e) => setForgotEmail(e.target.value)}
            disabled
          />

          {!isForgotOTPSent && (
            <button
              type="button"
              className="auth-form-btn secondary"
              onClick={handleForgotSendOTP}
            >
              Send OTP
            </button>
          )}

          {isForgotOTPSent && !isForgotOTPVerified && (
            <>
              <label htmlFor="forgotOTP">Enter OTP</label>
              <input
                id="forgotOTP"
                type="text"
                placeholder="6-digit OTP"
                value={forgotOTP}
                onChange={(e) => setForgotOTP(e.target.value)}
              />
              <button
                type="button"
                className="auth-form-btn secondary"
                onClick={handleForgotVerifyOTP}
              >
                Verify OTP
              </button>
            </>
          )}

          <label htmlFor="forgotNewPassword">New Password</label>
          <input
            id="forgotNewPassword"
            type={forgotShowPassword ? 'text' : 'password'}
            placeholder="Enter your new password"
            value={forgotNewPassword}
            onChange={(e) => setForgotNewPassword(e.target.value)}
            disabled={!isForgotOTPVerified}
          />

          <label htmlFor="forgotConfirmPassword">Confirm Password</label>
          <input
            id="forgotConfirmPassword"
            type={forgotShowPassword ? 'text' : 'password'}
            placeholder="Re-enter your new password"
            value={forgotConfirmPassword}
            onChange={(e) => setForgotConfirmPassword(e.target.value)}
            disabled={!isForgotOTPVerified}
          />

          {/* Show Password Checkbox */}
          {isForgotOTPSent && (
            <div className="checkbox-container">
              <input
                type="checkbox"
                id="forgotShowPassword"
                checked={forgotShowPassword}
                onChange={(e) => setForgotShowPassword(e.target.checked)}
                disabled={!isForgotOTPSent}
              />
              <label htmlFor="forgotShowPassword">Show Password</label>
            </div>
          )}

          <button
            type="submit"
            className="auth-form-btn"
            disabled={!isForgotOTPVerified}
          >
            Reset Password
          </button>

          <div className="auth-form-footer">
            <p>
              Remembered your password?{' '}
              <span className="link" onClick={() => handleSwitchMode('login')}>
                Sign in
              </span>
            </p>
          </div>
        </form>
      )}
    </div>
  );
}

export default AuthForm;
