import { NextRequest } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

export async function POST(req: NextRequest) {
  try {
    const { prompt } = await req.json();

    if (!prompt || typeof prompt !== 'string') {
      return new Response(JSON.stringify({ error: 'Prompt is required' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // Resolve paths relative to the project root (parent of ui/)
    const projectRoot = path.resolve(process.cwd(), '..');
    
    // Allow overriding via environment variables, or fallback to default build paths
    const cliPath = process.env.INFER_SPORE_CLI || path.join(projectRoot, 'build', 'bin', 'inferspore_cli');
    const weightsDir = process.env.INFER_SPORE_WEIGHTS || path.join(projectRoot, 'weights');
    const tokenizerPath = process.env.INFER_SPORE_TOKENIZER || path.join(projectRoot, 'tokenizer.flm');

    // Check if the CLI exists
    if (!fs.existsSync(cliPath)) {
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ token: "CLI not found. " })}\n\n`));
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ token: `Please build the engine at ${cliPath} first.` })}\n\n`));
          controller.close();
        },
      });
      return new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } });
    }

    // Start the C++ subprocess
    const child = spawn(cliPath, [weightsDir, tokenizerPath, prompt]);

    const stream = new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder();

        child.stdout.on('data', (data: Buffer) => {
          // Send the raw text chunk as an SSE data packet
          const text = data.toString('utf-8');
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ token: text })}\n\n`));
        });

        child.stderr.on('data', (data: Buffer) => {
          console.error(`CLI Stderr: ${data.toString()}`);
        });

        child.on('close', (code) => {
          if (code !== 0) {
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({ error: `Process exited with code ${code}` })}\n\n`)
            );
          }
          controller.close();
        });
        
        child.on('error', (err) => {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ error: `Failed to start process: ${err.message}` })}\n\n`)
          );
          controller.close();
        });
      },
      cancel() {
        // If the client disconnects, kill the child process
        child.kill();
      }
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
      },
    });
  } catch (error: any) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
