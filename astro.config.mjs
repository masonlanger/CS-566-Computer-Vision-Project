// @ts-check
import { defineConfig } from 'astro/config';
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// https://astro.build/config
export default defineConfig({
	site: 'https://masonlanger.github.io',
	base: '/CS-566-Computer-Vision-Project',
	markdown: {
		syntaxHighlight: "prism",
		remarkPlugins: [remarkMath],
    	rehypePlugins: [rehypeKatex],
	}
});
