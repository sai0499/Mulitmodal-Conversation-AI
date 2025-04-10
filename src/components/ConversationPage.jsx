import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './ConversationPage.css';

// Import icons from react-icons
import { CgAdd, CgLogOut } from 'react-icons/cg';
import { IoArrowUpCircle } from 'react-icons/io5';
import { VscGlobe } from 'react-icons/vsc';
import { FaRegCircleStop } from 'react-icons/fa6';
import { FiMic } from 'react-icons/fi';
import { TiWeatherSunny } from 'react-icons/ti';
import { PiMoonStarsFill } from 'react-icons/pi';
import { LuPenLine, LuSearch, LuPanelRightOpen, LuPanelLeftOpen } from "react-icons/lu";
import { MdOutlineDeleteForever } from "react-icons/md";

export default function ConversationPage() {
  // State declarations
  const [matricNumber, setMatricNumber] = useState('');
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [audioChunks, setAudioChunks] = useState([]);
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [visualiserData, setVisualiserData] = useState([]);
  const [darkMode, setDarkMode] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState(null);

  // For web search mode in chat area
  const [searchMode, setSearchMode] = useState(false);
  // Sidebar state
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  // State for chat history search (sidebar)
  const [isChatSearchActive, setIsChatSearchActive] = useState(false);
  const [chatSearchQuery, setChatSearchQuery] = useState('');
  // State for confirmation modal (for delete conversation)
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [chatToDelete, setChatToDelete] = useState(null);

  // Refs for audio, messages, textarea etc.
  const mediaRecorderRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const textAreaRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameIdRef = useRef(null);

  const navigate = useNavigate();

  // Set matric number from localStorage on mount; if absent, redirect to login page.
  useEffect(() => {
    const storedMatric = localStorage.getItem('matricNumber');
    if (storedMatric) {
      setMatricNumber(storedMatric);
    } else {
      navigate('/');
    }
  }, [navigate]);

  // Fetch user chat history from backend when component mounts.
  useEffect(() => {
    const fetchConversations = async () => {
      try {
        const token = localStorage.getItem('token');
        const response = await axios.get('http://localhost:4000/api/conversations', {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (response.data.success) {
          // Map conversations to a list to display in the sidebar.
          const history = response.data.conversations.map(conv => ({
            id: conv._id,
            title: conv.messages.find(m => m.sender === 'user')?.text.substring(0, 20) || 'New Chat'
          }));
          setChatHistory(history);
        }
      } catch (error) {
        console.error('Error fetching conversation history:', error);
      }
    };
    fetchConversations();
  }, []);

  // Toggle dark mode on document body.
  useEffect(() => {
    if (darkMode) {
      document.body.classList.add('dark-mode');
    } else {
      document.body.classList.remove('dark-mode');
    }
  }, [darkMode]);

  // Auto-scroll messages container on new messages.
  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // Auto-resize textarea (limit to 5 lines)
  const adjustTextAreaHeight = () => {
    if (textAreaRef.current) {
      textAreaRef.current.style.height = 'auto';
      const scrollHeight = textAreaRef.current.scrollHeight;
      const computedStyle = window.getComputedStyle(textAreaRef.current);
      const lineHeight = parseFloat(computedStyle.lineHeight) || 24;
      const maxHeight = lineHeight * 5;
      if (scrollHeight > maxHeight) {
        textAreaRef.current.style.height = `${maxHeight}px`;
        textAreaRef.current.style.overflowY = 'auto';
      } else {
        textAreaRef.current.style.height = `${scrollHeight}px`;
        textAreaRef.current.style.overflowY = 'hidden';
      }
    }
  };

  // Remove an attached file
  const removeFile = (index) => {
    setAttachedFiles(prev => prev.filter((_, i) => i !== index));
  };

  // Update audio visualiser using Web Audio API
  const updateVisualiser = () => {
    if (analyserRef.current) {
      const bufferLength = analyserRef.current.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyserRef.current.getByteFrequencyData(dataArray);
      const numBars = 73;
      const barData = [];
      const chunkSize = Math.floor(bufferLength / numBars);
      for (let i = 0; i < numBars; i++) {
        let sum = 0;
        for (let j = 0; j < chunkSize; j++) {
          sum += dataArray[i * chunkSize + j];
        }
        barData.push((sum / chunkSize) * 1.5);
      }
      setVisualiserData(barData);
      animationFrameIdRef.current = requestAnimationFrame(updateVisualiser);
    }
  };

  // Handle key down for Enter (without Shift) to send message.
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Logout action: remove token and matric number, then redirect.
  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('matricNumber');
    navigate('/');
  };

  // Toggle dark mode.
  const toggleDarkMode = () => {
    setDarkMode(prev => !prev);
  };

  // Handle file upload.
  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAttachedFiles(prev => [...prev, file]);
    try {
      const formData = new FormData();
      formData.append('file', file);
      await axios.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      alert(`File "${file.name}" uploaded successfully!`);
    } catch (err) {
      console.error('File upload error:', err);
      alert('Error uploading file.');
    }
  };

  // Toggle the existing web search mode in the chat area.
  const toggleSearch = async () => {
    const newValue = !searchMode;
    setSearchMode(newValue);
    try {
      await axios.post('/api/search-toggle', { searchMode: newValue });
    } catch (err) {
      console.error('Search toggle error:', err);
    }
  };

  // Toggle chat history search in the sidebar.
  const toggleChatSearch = () => {
    setIsChatSearchActive(prev => !prev);
    setChatSearchQuery('');
  };

  // Send message handler with token authentication and conversation grouping.
  // If the conversation is new, update the chat history immediately.
  const handleSendMessage = async () => {
    if (userInput.trim().length === 0) return;

    // Append the user message to UI immediately.
    const userMsg = {
      sender: 'user',
      text: userInput,
      files: attachedFiles.map(file => ({ name: file.name })),
    };
    setMessages(prev => [...prev, userMsg]);

    const textToSend = userInput;
    setUserInput('');
    setAttachedFiles([]);
    if (textAreaRef.current) {
      textAreaRef.current.style.height = 'auto';
    }
    setIsSending(true);

    try {
      const token = localStorage.getItem('token');
      const { data } = await axios.post(
        'http://localhost:4000/api/conversation',
        { text: textToSend, conversationId: activeConversationId },
        { headers: { Authorization: `Bearer ${token}` } }
      );

      // If conversation is newly created, store its ID and update chat history.
      if (!activeConversationId && data.conversationId) {
        setActiveConversationId(data.conversationId);
        setChatHistory(prev => ([
          {
            id: data.conversationId,
            title: textToSend.substring(0, 20) || 'New Chat'
          },
          ...prev
        ]));
      }

      setMessages(prev => [
        ...prev,
        { sender: 'bot', text: data.reply },
      ]);
    } catch (err) {
      console.error('Error sending message:', err);
      setMessages(prev => [
        ...prev,
        { sender: 'bot', text: 'Error in response.' },
      ]);
    }
    setIsSending(false);
  };

  // Instead of using window.confirm, set the chat to delete and show our custom confirmation modal.
  const handleDeleteConversation = (chatId) => {
    setChatToDelete(chatId);
    setShowConfirmModal(true);
  };

  // Called when the user confirms deletion in the modal.
  const confirmDeleteConversation = async () => {
    try {
      const token = localStorage.getItem('token');
      await axios.delete(`http://localhost:4000/api/conversation/${chatToDelete}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      // Remove conversation from chatHistory.
      setChatHistory(prev => prev.filter(chat => chat.id !== chatToDelete));
      // If the active conversation is deleted, clear messages and activeConversationId.
      if (activeConversationId === chatToDelete) {
        setActiveConversationId(null);
        setMessages([]);
      }
    } catch (error) {
      console.error("Error deleting conversation:", error);
      alert("Error deleting conversation.");
    } finally {
      setShowConfirmModal(false);
      setChatToDelete(null);
    }
  };

  // Hide the confirmation modal without deleting.
  const cancelDeleteConversation = () => {
    setShowConfirmModal(false);
    setChatToDelete(null);
  };

  // Start voice recording using Web Audio API.
  const handleStartRecording = async () => {
    setIsRecording(true);
    setAudioChunks([]);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          setAudioChunks(prev => [...prev, event.data]);
        }
      };
      mediaRecorder.start();

      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;
      source.connect(analyser);
      updateVisualiser();
    } catch (err) {
      console.error('Microphone access error:', err);
      alert('Please grant permission to use the Microphone');
      setIsRecording(false);
    }
  };

  // Stop voice recording.
  const handleStopRecording = () => {
    setIsRecording(false);
    if (animationFrameIdRef.current) {
      cancelAnimationFrame(animationFrameIdRef.current);
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    setVisualiserData([]);
    if (!mediaRecorderRef.current) return;
    mediaRecorderRef.current.stop();
    mediaRecorderRef.current.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      try {
        await axios.post('/api/upload-audio', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        alert('Audio clip successfully sent to backend!');
      } catch (err) {
        console.error('Audio upload error:', err);
      }
    };
  };

  // Determine if send button should be disabled.
  const isSendDisabled = isRecording || isSending || userInput.trim().length === 0;

  // Sidebar actions: clear current conversation (start a new chat).
  const handleNewChat = () => {
    setMessages([]);
    setActiveConversationId(null);
    console.log('New chat initiated');
  };

  // Load full conversation when a chat history item is clicked.
  const handleSelectChat = async (chatId) => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(`http://localhost:4000/api/conversation/${chatId}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (response.data.success) {
        setActiveConversationId(chatId);
        setMessages(response.data.conversation.messages);
      }
    } catch (error) {
      console.error("Error fetching conversation: ", error);
    }
  };

  // Filter chat history based on the search query.
  const filteredChatHistory = chatHistory.filter(chat =>
    chat.title.toLowerCase().includes(chatSearchQuery.toLowerCase())
  );

  return (
    <div className={`conversation-page ${darkMode ? 'dark-mode' : ''}`}>
      {/* Sidebar */}
      <div className={`sidebar ${isSidebarOpen ? 'open' : 'closed'}`}>
        {isSidebarOpen && (
          <>
            <div className="sidebar-header">
              <div className="sidebar-left">
                <button
                  className="collapse-sidebar-btn"
                  title="Collapse Sidebar"
                  onClick={() => setIsSidebarOpen(false)}
                >
                  <LuPanelRightOpen className="icon" />
                </button>
              </div>
              <div className="sidebar-right">
                <button
                  className="search-sidebar-btn"
                  title="Search Chats"
                  onClick={toggleChatSearch}
                >
                  <LuSearch className="icon" />
                </button>
                <button
                  className="new-chat-btn"
                  title="New Chat"
                  onClick={handleNewChat}
                >
                  <LuPenLine className="icon" />
                </button>
              </div>
            </div>
            <div className="sidebar-content">
              {isChatSearchActive && (
                <div className="chat-search-container">
                  <input
                    type="text"
                    placeholder="Search chats..."
                    value={chatSearchQuery}
                    onChange={(e) => setChatSearchQuery(e.target.value)}
                    className="chat-search-input"
                  />
                </div>
              )}
              {filteredChatHistory.length > 0 ? (
                <>
                  <h2 className="chat-history-title">Chat History</h2>
                  <ul className="chat-history">
                    {chatHistory.map((chat) => (
                      <li key={chat.id} className="chat-item">
                        <div
                          className="conversation-title"
                          onClick={() => handleSelectChat(chat.id)}
                        >
                          {chat.title}
                        </div>
                        <button
                          onClick={() => handleDeleteConversation(chat.id)}
                          title="Delete Conversation"
                          style={{
                            background: 'none',
                            border: 'none',
                            padding: 0,
                            cursor: 'pointer'
                          }}
                        >
                          <MdOutlineDeleteForever className="icon delete-icon" />
                        </button>
                      </li>
                    ))}
                  </ul>
                </>
              ) : (
                <p style={{ padding: '10px 20px' }}>No chats found</p>
              )}
            </div>
          </>
        )}
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="top-nav">
          <div className="nav-left">
            {!isSidebarOpen ? (
              <div className="header-row">
                <button
                  className="collapse-sidebar-btn"
                  title="Expand Sidebar"
                  onClick={() => setIsSidebarOpen(true)}
                >
                  <LuPanelLeftOpen className="icon" />
                </button>
                <button
                  className="new-chat-btn"
                  title="New Chat"
                  onClick={handleNewChat}
                >
                  <LuPenLine className="icon" />
                </button>
                <h1 className="app-title">UNI-ASK</h1>
              </div>
            ) : (
              <h1 className="app-title">UNI-ASK</h1>
            )}
          </div>
          <div className="nav-right">
            <span className="matric-display">
              Matriculation Number:&nbsp;<strong>{matricNumber}</strong>
            </span>
            <button
              className="dark-mode-btn"
              title="Toggle Dark Mode"
              onClick={toggleDarkMode}
            >
              {darkMode ? (
                <PiMoonStarsFill className="icon dark-mode-icon" />
              ) : (
                <TiWeatherSunny className="icon dark-mode-icon" />
              )}
            </button>
            <button className="logout-btn" title="Logout" onClick={handleLogout}>
              <CgLogOut className="icon logout-icon" />
            </button>
          </div>
        </div>

        {/* Chat Area */}
        <div className="chat-area">
          {messages.length > 0 && (
            <div className="messages-container" ref={messagesContainerRef}>
              {messages.map((msg, idx) => (
                <div key={idx} className={`message-row ${msg.sender === 'user' ? 'user' : 'bot'}`}>
                  <div className="message-text">
                    {msg.text}
                    {msg.files && msg.files.length > 0 && (
                      <div className="attached-files-message">
                        {msg.files.map((file, index) => (
                          <div key={index} className="attached-file-message">
                            <span className="file-name">{file.name}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Input Bar */}
          <div className={`input-bar ${messages.length > 0 ? 'bottom' : 'centered'}`}>
            {attachedFiles.length > 0 && (
              <div className="attached-files">
                {attachedFiles.map((file, idx) => (
                  <div key={idx} className="attached-file">
                    <span className="file-name">{file.name}</span>
                    <button className="remove-file" onClick={() => removeFile(idx)}>Remove</button>
                  </div>
                ))}
              </div>
            )}
            {!isRecording && (
              <textarea
                ref={textAreaRef}
                className="user-input"
                placeholder="Ask anything..."
                value={userInput}
                onChange={(e) => {
                  setUserInput(e.target.value);
                  adjustTextAreaHeight();
                }}
                onKeyDown={handleKeyDown}
                rows={1}
                title="Type your message"
              />
            )}
            <div className="buttons-row">
              <div className="left-controls">
                <label className="file-label" htmlFor="file-input" title="Add a file (PDF/TXT/JPG)">
                  <CgAdd className="icon plus-icon" />
                  <input
                    id="file-input"
                    type="file"
                    accept=".pdf,.txt,.jpeg,.jpg"
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                  />
                </label>
                <button
                  className={`search-toggle ${searchMode ? 'active' : ''}`}
                  title="Toggle search mode"
                  onClick={toggleSearch}
                >
                  <VscGlobe className="icon search-icon" />
                  <span className="search-text">Web Search</span>
                </button>
              </div>
              <div className="right-controls">
                {isRecording && (
                  <div className="audio-visualiser">
                    {visualiserData.map((value, index) => (
                      <div key={index} className="visualiser-bar" style={{ height: `${Math.min((value / 255) * 150, 100)}%` }}></div>
                    ))}
                  </div>
                )}
                <button
                  className="mic-btn"
                  title={isRecording ? 'Stop recording' : 'Start recording'}
                  onClick={isRecording ? handleStopRecording : handleStartRecording}
                  disabled={userInput.trim().length > 0}
                >
                  {isRecording ? (
                    <FaRegCircleStop className="icon stop-icon" />
                  ) : (
                    <FiMic className="icon mic-icon" />
                  )}
                </button>
                <button
                  className="send-btn"
                  title="Send message"
                  onClick={handleSendMessage}
                  disabled={isSendDisabled}
                >
                  <IoArrowUpCircle className="icon send-icon" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Custom Confirmation Modal */}
      {showConfirmModal && (
        <div className="confirm-modal-overlay">
          <div className="confirm-modal">
            <h3>Confirm Delete</h3>
            <p>Are you sure you want to delete this conversation?</p>
            <div>
              <button className="confirm-btn" onClick={confirmDeleteConversation}>Delete</button>
              <button className="cancel-btn" onClick={cancelDeleteConversation}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
