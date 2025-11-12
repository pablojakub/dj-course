import {GoogleGenAI} from "@google/genai";
import * as dotenv from 'dotenv';

dotenv.config();

const API_KEY = process.env.GEMINI_API_KEY;

const maskedKey = API_KEY
    ? API_KEY.substring(0, 4) + '...' + API_KEY.substring(API_KEY.length - 4)
    : 'NOT SET';
console.log(`env var "GEMINI_API_KEY" is: ${maskedKey}`);

if (!API_KEY) {
    throw new Error("GEMINI_API_KEY environment variable is not set. Please set it to your Google Gemini API key.");
}

const ai = new GoogleGenAI({});

const model = "gemini-2.5-flash";
// const model = "gemini-2.5-pro";

const systemRole = "You are an evil prank-addict. You’re never serious";

const conversationHistory = [
    {
        role: "user",
        parts: [{text: "What is the best time for coffee?"}]
    },
    {
        role: "model",
        parts: [{text: "At night with owls and dead bodies"}]
    },
    {
        role: "user",
        parts: [{text: "How about tea?"}]
    },
];

async function run() {
    try {
        console.log("\nSending query to Gemini...");

        const response = await ai.models.generateContent({
            model: model,
            contents: conversationHistory,
            config: {
                systemInstruction: systemRole,
                thinkingConfig: {
                    thinkingBudget: 0
                }
            },
        });

        console.log("Gandalf's answer:");
        console.log(response.text);

    } catch (error) {
        console.error("An error occurred:", error.message);
    }
}

run();
