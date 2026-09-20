import "./globals.css";

export const metadata = {
  title: "Cadabra",
  description: "A 4B model post-trained on Baseten reads an engineering drawing and writes the CAD program that builds the part.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head><link rel="icon" href="data:," /></head>
      <body>{children}</body>
    </html>
  );
}
