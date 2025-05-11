import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './ConversationPage.css';
import FileAttachment from './FileAttachment';

// Import icons from react-icons
import { CgAdd, CgLogIn } from 'react-icons/cg';
import { IoArrowUpCircle } from 'react-icons/io5';
import { VscGlobe } from 'react-icons/vsc';
import { FaRegCircleStop } from 'react-icons/fa6';
import { FiMic } from 'react-icons/fi';
import { TiWeatherSunny } from 'react-icons/ti';
import { PiMoonStarsFill } from 'react-icons/pi';
import { BiCopy } from 'react-icons/bi';
import { IoMdVolumeHigh } from 'react-icons/io';
import { TbApi } from 'react-icons/tb';
import { FaCheckCircle } from 'react-icons/fa';

import {
  LuPenLine,
  LuSearch,
  LuPanelRightOpen,
  LuPanelLeftOpen,
} from 'react-icons/lu';
import { MdOutlineDeleteForever } from 'react-icons/md';
import MarkdownRenderer from './MarkdownRenderer';

export default function ConversationPage() {
  // State declarations
  const [matricNumber, setMatricNumber] = useState('');
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [visualiserData, setVisualiserData] = useState([]);
  const [darkMode, setDarkMode] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState(null);

  const [userApiKey, setUserApiKey] = useState('');
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [newApiKey, setNewApiKey] = useState('');

  const [apiKeySaved, setApiKeySaved] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

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

  const [tempPopupMessage, setTempPopupMessage] = useState('');
  const [showTempPopup, setShowTempPopup] = useState(false);

  // Refs for audio recording, messages, etc.
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const messagesContainerRef = useRef(null);
  const textAreaRef = useRef(null);
  const audioContextRef = useRef(null);
  const animationFrameIdRef = useRef(null);
  const chatSearchRef = useRef(null);

  const [ttsActiveMessageId, setTTSActiveMessageId] = useState(null);
  const ttsUtteranceRef = useRef(null);
  const [copiedMessageIndex, setCopiedMessageIndex] = useState(null);

  const navigate = useNavigate();

  // On mount, set matric number; if absent, redirect.
  useEffect(() => {
    const storedMatric = sessionStorage.getItem('matricNumber');
    if (storedMatric) {
      setMatricNumber(storedMatric);
    } else {
      navigate('/');
    }
  }, [navigate]);

  // On mount, fetch the user’s API key from the backend.
  useEffect(() => {
    const fetchApiKey = async () => {
      try {
        const token = sessionStorage.getItem('token');
        const response = await axios.get('http://localhost:4000/api/user/apiKey', {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (response.data.success && response.data.apiKey) {
          setUserApiKey(response.data.apiKey);
        }
      } catch (error) {
        console.error('Error fetching API key:', error);
      }
    };
    fetchApiKey();
  }, []);

  // On mount, load activeConversationId from localStorage and fetch its messages
  useEffect(() => {
    const storedActiveConversationId = sessionStorage.getItem('activeConversationId');
    if (storedActiveConversationId) {
      setActiveConversationId(storedActiveConversationId);
      fetchConversation(storedActiveConversationId);
    }
  }, []);

  // Helper function: Fetch conversation details by ID.
  const fetchConversation = async (conversationId) => {
    try {
      const token = sessionStorage.getItem('token');
      const response = await axios.get(
        `http://localhost:4000/api/conversation/${conversationId}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      if (response.data.success) {
        setMessages(response.data.conversation.messages);
      }
    } catch (error) {
      console.error('Error fetching conversation:', error);
    }
  };

  const handleCopy = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedMessageIndex(idx);
    // Clear the copy status after 3 seconds
    setTimeout(() => setCopiedMessageIndex(null), 3000);
  };

  const handleToggleTTS = (idx, text) => {
    if (ttsActiveMessageId === idx) {
      // If already reading this message, cancel.
      speechSynthesis.cancel();
      setTTSActiveMessageId(null);
    } else {
      // Stop any ongoing speech.
      speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      ttsUtteranceRef.current = utterance;
      speechSynthesis.speak(utterance);
      setTTSActiveMessageId(idx);

      // When TTS ends, clear the state.
      utterance.onend = () => {
        setTTSActiveMessageId(null);
      };
    }
  };
  // Reset tts active state & Cancel any ongoing speech
  const stopTTS = () => {
    speechSynthesis.cancel();
    setTTSActiveMessageId(null);
  };

  // Update handleSelectChat to also persist the selected conversation.
  const handleSelectChat = async (chatId) => {
    try {
      stopTTS();
      const token = sessionStorage.getItem('token');
      const response = await axios.get(
        `http://localhost:4000/api/conversation/${chatId}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      if (response.data.success) {
        setActiveConversationId(chatId);
        sessionStorage.setItem('activeConversationId', chatId);
        setMessages(response.data.conversation.messages);
      }
    } catch (error) {
      console.error('Error fetching conversation: ', error);
    }
  };

  // Fetch conversation history.
  useEffect(() => {
    const fetchConversations = async () => {
      try {
        const token = sessionStorage.getItem('token');
        const response = await axios.get('http://localhost:4000/api/conversations', {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (response.data.success) {
          const history = response.data.conversations.map((conv) => ({
            id: conv._id,
            // Use stored title if available; otherwise, fallback.
            title:
              conv.title ||
              (conv.messages.find((m) => m.sender === 'user')?.text.substring(0, 20) || 'New Chat'),
            loading: false,
          }));
          setChatHistory(history);
        }
      } catch (error) {
        console.error('Error fetching conversation history:', error);
      }
    };
    fetchConversations();
  }, []);

  // Automatically focus the search input when search is activated.
  useEffect(() => {
    if (isChatSearchActive && chatSearchRef.current) {
      chatSearchRef.current.focus();
    }
  }, [isChatSearchActive]);

  // On mount, load dark mode preference
  useEffect(() => {
    const storedDarkMode = localStorage.getItem('darkMode');
    if (storedDarkMode === "true") {
      setDarkMode(true);
    }
  }, []);

  // When darkMode changes, update localStorage and document body class
  useEffect(() => {
    localStorage.setItem('darkMode', darkMode);
    if (darkMode) {
      document.body.classList.add('dark-mode');
    } else {
      document.body.classList.remove('dark-mode');
    }
  }, [darkMode]);

  // Auto-scroll messages container.
  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // Auto-resize textarea.
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

  useEffect(() => {
    adjustTextAreaHeight();
  }, [userInput]);

  // Update audio visualiser.
  const updateVisualiser = (analyser) => {
    if (analyser) {
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyser.getByteFrequencyData(dataArray);
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
      animationFrameIdRef.current = requestAnimationFrame(() => updateVisualiser(analyser));
    }
  };

  // Handle Enter key.
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Logout action.
  const handleLogout = () => {
    stopTTS();
    sessionStorage.removeItem('token');
    sessionStorage.removeItem('matricNumber');
    sessionStorage.removeItem('activeConversationId');
    navigate('/');
  };

  // Toggle dark mode.
  const toggleDarkMode = () => {
    setDarkMode((prev) => !prev);
  };

  // File upload handler.
  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAttachedFiles((prev) => [...prev, file]);
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

  // Remove a file.
  const removeFile = (index) => {
    setAttachedFiles((prev) => {
      const updated = [...prev];
      const fileToRemove = updated[index];
      if (fileToRemove && fileToRemove.previewUrl) {
        URL.revokeObjectURL(fileToRemove.previewUrl);
      }
      updated.splice(index, 1);
      return updated;
    });
  };

  /**
* Shows a popup message that disappears after 2 seconds.
*/
  const showTemporaryPopup = (message) => {
    setTempPopupMessage(message);
    setShowTempPopup(true);

    setTimeout(() => {
      setShowTempPopup(false);
      setTempPopupMessage('');
    }, 2000); // 2 seconds
  };


  // Toggle web search mode.
  const toggleSearch = async () => {
    if (!userApiKey) {
      showTemporaryPopup('⚠️ No API key is added for online search.');
      return;
    }
    const newValue = !searchMode;
    setSearchMode(newValue);
    try {
      await axios.post('http://localhost:5000/api/search-toggle', { searchMode: newValue });
    } catch (err) {
      console.error('Search toggle error:', err);
    }
  };

  // Toggle sidebar chat search.
  const toggleChatSearch = () => {
    setIsChatSearchActive((prev) => !prev);
    setChatSearchQuery('');
  };

  // Helper function: simulate typing animation for bot responses.
  const simulateTyping = (fullText) => {
    return new Promise((resolve) => {
      // Remove the blinking effect as soon as typing simulation begins.
      setMessages((prevMessages) => {
        const newMessages = [...prevMessages];
        if (newMessages.length > 0 && newMessages[newMessages.length - 1].loading) {
          newMessages[newMessages.length - 1].loading = false;
        }
        return newMessages;
      });

      let index = 0;
      const typingInterval = setInterval(() => {
        index++;
        setMessages((prevMessages) => {
          const newMessages = [...prevMessages];
          if (newMessages.length > 0) {
            newMessages[newMessages.length - 1].text = fullText.substring(0, index);
          }
          return newMessages;
        });
        if (index >= fullText.length) {
          clearInterval(typingInterval);
          resolve();
        }
      }, 10); // Adjust the delay as needed for a natural typing speed.
    });
  };

  // Send message handler.
  const handleSendMessage = async () => {
    stopTTS();
    if (userInput.trim().length === 0) return;
    // Immediately append user message.
    const userMsg = {
      sender: 'user',
      text: userInput,
      files: attachedFiles.map((file) => ({ name: file.name })),
    };
    setMessages((prev) => [...prev, userMsg]);

    const textToSend = userInput;
    setUserInput('');
    setAttachedFiles([]);
    if (textAreaRef.current) {
      textAreaRef.current.style.height = 'auto';
    }
    setIsSending(true);

    // If new conversation, add a temporary sidebar item.
    if (!activeConversationId) {
      setChatHistory((prev) => [
        {
          id: 'temp',
          title: '.....',
          loading: true,
        },
        ...prev,
      ]);
    }


    // Insert placeholder bot message.
    const placeholder = { sender: 'bot', text: 'generating response...', loading: true };
    setMessages((prev) => [...prev, placeholder]);

    try {
      const token = sessionStorage.getItem('token');
      const formData = new FormData();
      formData.append('text', textToSend);
      if (activeConversationId) {
        formData.append('conversationId', activeConversationId);
      }
      attachedFiles.forEach((file) => {
        formData.append('files', file);
      });
      const { data } = await axios.post(
        'http://localhost:4000/api/conversation',
        formData,
        { headers: { Authorization: `Bearer ${token}` } }
      );

      // For new conversation, update temporary sidebar item.
      if (!activeConversationId && data.conversationId) {
        setActiveConversationId(data.conversationId);
        sessionStorage.setItem('activeConversationId', data.conversationId);
        setChatHistory((prev) => {
          const filtered = prev.filter((item) => item.id !== 'temp');
          return [
            {
              id: data.conversationId,
              title: data.title || (textToSend.substring(0, 20) || 'New Chat'),
              loading: false,
            },
            ...filtered,
          ];
        });
      }

      // Animate typing for bot reply.
      await simulateTyping(data.reply);

      // Remove loading flag from bot message.
      setMessages((prev) => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1] = {
          sender: 'bot',
          text: newMessages[newMessages.length - 1].text,
          loading: false,
        };
        return newMessages;
      });
    } catch (err) {
      console.error('Error sending message:', err);
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          sender: 'bot',
          text: 'Error in response.',
          loading: false,
        };
        return updated;
      });
    }
    setIsSending(false);
  };

  // Delete conversation actions.
  const handleDeleteConversation = (chatId) => {
    setChatToDelete(chatId);
    setShowConfirmModal(true);
  };

  const confirmDeleteConversation = async () => {
    try {
      const token = sessionStorage.getItem('token');
      await axios.delete(`http://localhost:4000/api/conversation/${chatToDelete}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setChatHistory((prev) =>
        prev.filter((chat) => chat.id !== chatToDelete)
      );
      if (activeConversationId === chatToDelete) {
        setActiveConversationId(null);
        setMessages([]);
      }
    } catch (error) {
      console.error('Error deleting conversation:', error);
      alert('Error deleting conversation.');
    } finally {
      setShowConfirmModal(false);
      setChatToDelete(null);
    }
  };

  const cancelDeleteConversation = () => {
    setShowConfirmModal(false);
    setChatToDelete(null);
  };

  // Start voice recording.
  const handleStartRecording = async () => {
    setIsRecording(true);
    audioChunksRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const preferredMime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/wav')
          ? 'audio/wav'
          : '';
      const options = preferredMime ? { mimeType: preferredMime } : undefined;
      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onerror = (event) => {
        console.error('MediaRecorder error:', event.error);
      };

      mediaRecorder.start();

      // Set up audio visualiser.
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      updateVisualiser(analyser);
    } catch (err) {
      console.error('Microphone access error:', err);
      alert('Please grant permission to use the Microphone');
      setIsRecording(false);
    }
  };

  // Stop recording and send audio for transcription.
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

    mediaRecorderRef.current.onstop = async () => {
      const audioData = audioChunksRef.current;
      const chunkType = audioData[0]?.type || 'audio/webm';
      const extension = chunkType.includes('wav') ? 'wav' : 'webm';
      const audioBlob = new Blob(audioData, { type: chunkType });
      console.log('Audio Blob size:', audioBlob.size);
      if (audioBlob.size === 0) {
        console.error('Error: Captured audio is empty. Please try recording again.');
        return;
      }
      const formData = new FormData();
      formData.append('audio', audioBlob, `recording.${extension}`);
      try {
        const response = await axios.post('http://localhost:5000/api/transcribe', formData);
        if (response.data.transcription) {
          setUserInput((prev) => prev + ' ' + response.data.transcription);
        }
      } catch (err) {
        console.error('Transcription error:', err);
      }
    };

    mediaRecorderRef.current.stop();
  };

  // Determine if send button should be disabled.
  const isSendDisabled = isRecording || isSending || userInput.trim().length === 0;

  // Sidebar actions.
  const handleNewChat = () => {
    stopTTS();
    setMessages([]);
    setActiveConversationId(null);
    // Clear active conversation from localStorage so that refresh stays on new chat.
    sessionStorage.removeItem('activeConversationId');
    console.log('New chat initiated');
  };

  // Filter chat history based on search query.
  const filteredChatHistory = chatHistory.filter((chat) =>
    chat.title.toLowerCase().includes(chatSearchQuery.toLowerCase())
  );

  const handleSaveApiKey = async () => {
    if (!newApiKey.trim()) {
      setErrorMessage('API Key cannot be empty.');
      return;
    }
    try {
      const token = sessionStorage.getItem('token');
      const response = await axios.post(
        'http://localhost:4000/api/user/apiKey',
        { apiKey: newApiKey },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      if (response.data.success) {
        setUserApiKey(response.data.apiKey);
        setNewApiKey('');
        setErrorMessage('');
        // Set success state to true to display the green tick
        setApiKeySaved(true);
        // Close modal after 2 seconds and clear the success indicator
        setTimeout(() => {
          setApiKeySaved(false);
          setShowApiKeyModal(false);
        }, 2000);
      } else {
        setErrorMessage('Error updating API key: ' + response.data.message);
      }
    } catch (error) {
      console.error('Error updating API key:', error);
      setErrorMessage('Error updating API key');
    }
  };
  // ------------------------------------------------

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
                <button className="new-chat-btn" title="New Chat" onClick={handleNewChat}>
                  <LuPenLine className="icon" />
                </button>
              </div>
            </div>
            <div className="sidebar-content">
              <div className="sidebar-logo-title">
                <img src="../favicon.svg" alt="Logo" className="sidebar-logo" />
                <span className="sidebar-title">Uni-Ask</span>
              </div>
              {isChatSearchActive && (
                <div className="chat-search-container">
                  <input
                    ref={chatSearchRef}
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
                    {filteredChatHistory.map((chat) => (
                      <li
                        key={chat.id}
                        className={`chat-item ${chat.id === activeConversationId ? 'active-chat' : ''}`}
                        onClick={() => handleSelectChat(chat.id)}
                      >
                        <div className="conversation-title">
                          {chat.title}
                        </div>
                        <button
                          onClick={() => handleDeleteConversation(chat.id)}
                          title="Delete Conversation"
                          style={{
                            background: 'none',
                            border: 'none',
                            padding: 0,
                            cursor: 'pointer',
                          }}
                        >
                          <MdOutlineDeleteForever className="icon delete-icon" />
                        </button>
                      </li>
                    ))}
                  </ul>
                </>
              ) : (
                <p style={{ padding: '10px 10px' }}>No chats found</p>
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
                <button className="new-chat-btn" title="New Chat" onClick={handleNewChat}>
                  <LuPenLine className="icon" />
                </button>
                <h1 className="app-title">Uni-Ask Multimodal AI</h1>
              </div>
            ) : (
              <h1 className="app-title">Uni-Ask Multimodal AI</h1>
            )}
          </div>
          <div className="nav-right">
            <span className="matric-display">
              Matriculation Number:&nbsp;<strong>{matricNumber}</strong>
            </span>
            {/* API Key Button */}
            <button
              className="api-key-btn"
              title="Add API Key"
              onClick={() => setShowApiKeyModal(true)}
            >
              <TbApi className="icon" />
            </button>
            <button className="dark-mode-btn" title="Toggle to Dark/Light Mode" onClick={toggleDarkMode}>
              {darkMode ? (
                <TiWeatherSunny className="icon dark-mode-icon" />
              ) : (
                <PiMoonStarsFill className="icon dark-mode-icon" />
              )}
            </button>
            <button className="logout-btn" title="Logout" onClick={handleLogout}>
              <CgLogIn className="icon logout-icon" />
            </button>
          </div>
        </div>

        {showTempPopup && (
          <div className="temporary-popup">
            {tempPopupMessage}
          </div>
        )}

        {/* Chat Area */}
        <div className="chat-area">
          {/* Welcome message for a new chat */}
          {messages.length === 0 && (
            <div className="welcome-message">
              <p>Welcome to Uni-Ask! How can I assist you today?</p>
            </div>
          )}
          {messages.length > 0 && (
            <div className="messages-container" ref={messagesContainerRef}>
              {messages.map((msg, idx) => (
                <div key={idx} className={`message-row ${msg.sender === 'user' ? 'user' : 'bot'}`}>
                  <div
                    className={`message-text ${msg.loading ? 'loading-text' : ''} ${msg.sender === 'bot' ? 'bot-message-container' : ''}`}
                  >
                    {msg.sender === 'bot' ? (
                      <MarkdownRenderer content={msg.text} />
                    ) : (
                      msg.text
                    )}
                    {msg.sender === 'bot' && !msg.loading && (
                      <div className="bot-actions">
                        <span className="copy-icon" onClick={() => handleCopy(msg.text, idx)}>
                          <BiCopy />
                          {copiedMessageIndex === idx && (
                            <span className="copied-message">Copied!</span>
                          )}
                        </span>
                        <span
                          className={`speaker-icon ${ttsActiveMessageId === idx ? 'active' : ''}`}
                          onClick={() => handleToggleTTS(idx, msg.text)}
                        >
                          <IoMdVolumeHigh />
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Input Bar */}
          <div className={`input-bar ${messages.length > 0 ? 'bottom' : 'centered'} ${activeConversationId ? 'no-animation' : ''}`}>
            {attachedFiles.length > 0 && (
              <div className="attached-files">
                {attachedFiles.map((file, idx) => (
                  <FileAttachment
                    key={idx}
                    file={file}
                    removable={true}
                    onRemove={() => removeFile(idx)}
                  />
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
                <label className="file-label" htmlFor="file-input" title="Add a file (PDF/TXT/JPG/PNG)">
                  <CgAdd className="icon plus-icon" />
                  <input
                    id="file-input"
                    type="file"
                    accept=".pdf,.txt,.jpeg,.jpg,.png"
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
                      <div
                        key={index}
                        className="visualiser-bar"
                        style={{
                          height: `${Math.min((value / 255) * 150, 100)}%`,
                        }}
                      ></div>
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

      {/* Delete Conversation Confirmation Modal */}
      {showConfirmModal && (
        <div className="confirm-modal-overlay">
          <div className="confirm-modal">
            <h3>Confirm Delete</h3>
            <p>Are you sure you want to delete this conversation?</p>
            <div>
              <button className="confirm-btn" onClick={confirmDeleteConversation}>
                Delete
              </button>
              <button className="cancel-btn" onClick={cancelDeleteConversation}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* API Key Modal */}
      {showApiKeyModal && (
        <div className="confirm-modal-overlay">
          <div className="confirm-modal">
            {apiKeySaved ? (
              // Show success UI with a green tick and success message.
              <div className="api-key-success" style={{ textAlign: 'center' }}>
                <FaCheckCircle size={48} color="green" />
                <p style={{ marginTop: '10px' }}>API key updated successfully.</p>
              </div>
            ) : (
              <>
                <h3>Add API Key</h3>
                <input
                  type="text"
                  value={newApiKey}
                  onChange={(e) => {
                    setNewApiKey(e.target.value);
                    setErrorMessage('');
                  }}
                  placeholder="Enter your SerpAPI Key"
                  style={{ width: '90%', padding: '8px', marginBottom: '20px', borderRadius: '10px', border: '1px solid gray' }}
                />
                {errorMessage && (
                  <p className="error-message" style={{ color: 'red', marginBottom: '10px' }}>
                    {errorMessage}
                  </p>
                )}
                <div>
                  <button className="confirm-btn" onClick={handleSaveApiKey}>Save</button>
                  <button className="cancel-btn" onClick={() => setShowApiKeyModal(false)}>Cancel</button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

    </div>
  );
}
