const chatHistory = document.getElementById('chat-history');
const chatForm = document.getElementById('chat-form');
const promptInput = document.getElementById('prompt-input');
const sendBtn = document.getElementById('send-btn');

// Auto-resize textarea
promptInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    if(this.value.trim() === '') {
        sendBtn.disabled = true;
    } else {
        sendBtn.disabled = false;
    }
});

promptInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if(!sendBtn.disabled) {
            chatForm.dispatchEvent(new Event('submit'));
        }
    }
});

function appendMessage(role, content) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (role === 'bot') {
        // We will render markdown and add a cursor
        contentDiv.innerHTML = marked.parse(content) + '<span class="cursor-blink"></span>';
    } else {
        contentDiv.textContent = content;
    }
    
    msgDiv.appendChild(contentDiv);
    chatHistory.appendChild(msgDiv);
    
    // Scroll to bottom
    chatHistory.scrollTop = chatHistory.scrollHeight;
    
    return contentDiv;
}

function updateBotMessage(contentDiv, fullContent, isFinished = false) {
    let html = marked.parse(fullContent);
    if (!isFinished) {
        html += '<span class="cursor-blink"></span>';
    }
    contentDiv.innerHTML = html;
    
    // Auto-scroll as text comes in if we are near the bottom
    const isScrolledToBottom = chatHistory.scrollHeight - chatHistory.clientHeight <= chatHistory.scrollTop + 50;
    if(isScrolledToBottom) {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const prompt = promptInput.value.trim();
    if (!prompt) return;
    
    // Reset input
    promptInput.value = '';
    promptInput.style.height = 'auto';
    sendBtn.disabled = true;
    
    // Add user message
    appendMessage('user', prompt);
    
    // Create placeholder for bot message
    const botContentDiv = appendMessage('bot', '');
    let fullResponse = '';
    
    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt: prompt })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            // SSE messages are separated by \n\n
            const lines = buffer.split('\n\n');
            buffer = lines.pop(); // Keep the last incomplete chunk in the buffer
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const dataStr = line.slice(6);
                    try {
                        const data = JSON.parse(dataStr);
                        if (data.error) {
                            appendMessage('error', data.error + (data.stderr ? ': ' + data.stderr : ''));
                        } else if (data.token) {
                            fullResponse += data.token;
                            updateBotMessage(botContentDiv, fullResponse);
                        }
                    } catch (e) {
                        console.error('Error parsing SSE data:', e, dataStr);
                    }
                }
            }
        }
        
        // Done receiving
        updateBotMessage(botContentDiv, fullResponse, true);
        
    } catch (error) {
        console.error('Generation error:', error);
        appendMessage('error', 'Failed to connect to the engine. Make sure the backend is running.');
        updateBotMessage(botContentDiv, fullResponse, true); // Remove cursor
    } finally {
        sendBtn.disabled = false;
        promptInput.focus();
    }
});

// Initialize Send Button state
sendBtn.disabled = true;
