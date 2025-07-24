import pandas as pd
import os
import google.generativeai as genai
import time


os.environ['GOOGLE_API_KEY'] = "AIzaSyAaclfgYAqOR0eF2juAUErLofXpUI6cwjM"  # Replace with your API key
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def generate_article(summary,avg_length, language="Hindi"):
    """
    Generate a full article using Gemini's content generation based on a summary.
    :param summary: The summary of the article.
    :param language: Language of the output article.
    """
    # prompt = f"""
    # You are a creative and skilled news writer. Based on the given summary, write a detailed and engaging news article in {language}.
    
    # Summary: "{summary}"

    # Your task:
    # - Expand the summary into a full article with relevant details, adding context and the background of the story.
    # - Use a reader-friendly tone, ensuring it’s engaging.
    # - Avoid directly copying the summary. Instead, use it as inspiration to develop a complete story.
    # - Keep the article authentic and provide information in a clear, logical flow.

    # Start your response directly with the article body, written entirely in Hindi.
    # """
    prompt = f"""
    You are an assistant skilled in writing {language} news articles. Expand the following summary into a {language} article of approximately {avg_length} words. Summary: "{summary}".
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating article: {e}")
        return None


# Example usage in the main function
def main():
    input_file = "Datasetlink.xlsx"  # Path to your dataset file
    num = 1000
    avg_length = 400
    output_file = f"Generated_Articles_From_Summary_gemini_{num}.csv"  # Path to save the generated articles

    # Load the dataset
    df = pd.read_excel(input_file)
    df = df.head(num)  # Consider only the first 20 rows for demonstration
    # Create a list to store generated articles
    generated_data = []
    cnt = 0

    for _, row in df.iterrows():
        summary = row['summary']
        print(f"Processing summary: {summary[:50]}...")  # Display first 50 characters of the summary for reference

        try:
            # Generate a full article using the summary
            full_article = generate_article(summary,avg_length)
            print(f"Generated article: {full_article[:100]}...")  # Display first 100 characters of the generated article

            # Append results to the list
            generated_data.append({
                "summary": summary,
                "original_article": row['full_article'],
                "__ORIGINAL_ARTICLE_END__": "__ORIGINAL_ARTICLE_END__",
                "generated_article": full_article,
                "__GENERATED_ARTICLE_END__": "__GENERATED_ARTICLE_END__"
            })

            cnt += 1
            if cnt % 5 == 0:
                print("Sleeping for 40 seconds to avoid rate limiting...")
                time.sleep(1)

        except Exception as e:
            print(f"Error processing summary: {e}")

    # Save the generated articles to a CSV file
    pd.DataFrame(generated_data).to_csv(output_file, index=False, encoding='utf-8')
    print(f"Generated articles saved to {output_file}")


if __name__ == "__main__":
    main()
