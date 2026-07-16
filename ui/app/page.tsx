'use client';

import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';

interface Message {
  role: 'user' | 'bot' | 'system' | 'error';
  content: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'system',
      content: 'Welcome to **plow**. High-performance Rust inference engine initialized. Waiting for prompt...',
    },
  ]);
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const chatHistoryRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [messages]);

  // Auto-resize textarea
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setPrompt(e.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isGenerating && prompt.trim()) {
        handleSubmit(e as any);
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || isGenerating) return;

    const userPrompt = prompt.trim();
    setPrompt('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
    
    setMessages((prev) => [...prev, { role: 'user', content: userPrompt }, { role: 'bot', content: '' }]);
    setIsGenerating(true);

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: userPrompt }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      if (!response.body) {
        throw new Error('ReadableStream not supported in this browser.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            try {
              const data = JSON.parse(dataStr);
              if (data.error) {
                setMessages((prev) => {
                  const newMsgs = [...prev];
                  newMsgs[newMsgs.length - 1] = { role: 'error', content: data.error };
                  return newMsgs;
                });
              } else if (data.token) {
                setMessages((prev) => {
                  const newMsgs = [...prev];
                  const lastMsg = newMsgs[newMsgs.length - 1];
                  lastMsg.content += data.token;
                  return newMsgs;
                });
              }
            } catch (err) {
              console.error('Error parsing SSE data:', err, dataStr);
            }
          }
        }
      }
    } catch (error: any) {
      setMessages((prev) => [
        ...prev,
        { role: 'error', content: error.message || 'Failed to connect to the engine.' },
      ]);
    } finally {
      setIsGenerating(false);
      setTimeout(() => {
        textareaRef.current?.focus();
      }, 0);
    }
  };

  return (
    <div className="flex justify-center items-center h-screen overflow-hidden text-[#f8fafc]">
      <div className="background-mesh">
        <div className="mesh-blob blob-1"></div>
        <div className="mesh-blob blob-2"></div>
      </div>

      <div className="w-full max-w-4xl mx-auto flex flex-col h-[90vh] px-4 gap-6">
        {/* Header */}
        <header className="glass-panel rounded-2xl p-4 flex justify-between items-center shadow-lg">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-purple-400">
              plow
            </h1>
          </div>
          <div className="flex items-center gap-2 px-3 py-1 bg-white/5 rounded-full border border-white/10 text-sm font-medium">
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
            CUDA Engine Ready
          </div>
        </header>

        {/* Chat Container */}
        <main className="glass-panel flex-1 rounded-2xl flex flex-col overflow-hidden shadow-2xl relative">
          
          {/* Chat History */}
          <div
            ref={chatHistoryRef}
            className="flex-1 overflow-y-auto p-6 flex flex-col gap-6 scroll-smooth"
          >
            {messages.map((msg, idx) => {
              const isLastBot = idx === messages.length - 1 && msg.role === 'bot' && isGenerating;
              
              let bubbleClasses = 'max-w-[85%] rounded-2xl p-4 ';
              if (msg.role === 'user') {
                bubbleClasses += 'self-end bg-white/5 border border-white/10';
              } else if (msg.role === 'system') {
                bubbleClasses += 'self-center bg-indigo-500/10 border border-indigo-500/20 text-center text-sm text-indigo-200';
              } else if (msg.role === 'error') {
                bubbleClasses += 'self-start bg-red-500/10 border border-red-500/20 text-red-400';
              } else {
                bubbleClasses += 'self-start bg-indigo-500/10 border border-indigo-500/20';
              }

              return (
                <div key={idx} className={bubbleClasses}>
                  <div className="prose prose-invert prose-p:leading-relaxed prose-pre:bg-black/30 prose-pre:border prose-pre:border-white/10">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                    {isLastBot && <span className="cursor-blink"></span>}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Input Area */}
          <div className="p-4 border-t border-white/10 bg-black/20">
            <form onSubmit={handleSubmit} className="flex gap-3">
              <div className="relative flex-1 bg-white/5 border border-white/10 rounded-xl focus-within:border-indigo-400/50 focus-within:ring-1 focus-within:ring-indigo-400/50 transition-all">
                <textarea
                  ref={textareaRef}
                  value={prompt}
                  onChange={handleInput}
                  onKeyDown={handleKeyDown}
                  placeholder="Enter your prompt here..."
                  rows={1}
                  className="w-full bg-transparent p-4 pr-12 text-white placeholder-white/40 focus:outline-none resize-none max-h-48 overflow-y-auto"
                />
                <button
                  type="submit"
                  disabled={isGenerating || !prompt.trim()}
                  className="absolute right-3 bottom-3 p-2 text-indigo-400 hover:text-indigo-300 hover:bg-indigo-400/10 rounded-lg transition-colors disabled:opacity-50 disabled:hover:bg-transparent"
                >
                  <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="22" y1="2" x2="11" y2="13"></line>
                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                  </svg>
                </button>
              </div>
            </form>
            <div className="text-center mt-2 text-xs text-white/40">
              Press Enter to send (Shift+Enter for new line)
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
