import argparse
import mimetypes
import os
import time
from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash-image-preview"


def remix_images(
    image_paths: list[str],
    prompt: str,
    output_dir: str,
):
    """
    Remixes two images using the Google Generative AI model.

    Args:
        image_paths: A list of two paths to input images.
        prompt: The prompt for remixing the images.
        output_dir: Directory to save the remixed images.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)

    contents = _load_image_parts(image_paths)
    contents.append(genai.types.Part.from_text(text=prompt))

    generate_content_config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
    )

    print(f"Remixing with {len(image_paths)} images and prompt: {prompt}")

    stream = client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=contents,
        config=generate_content_config,
    )

    _process_api_stream_response(stream, output_dir)


def _load_image_parts(image_paths: list[str]) -> list[types.Part]:
    """Loads image files and converts them into GenAI Part objects."""
    parts = []
    for image_path in image_paths:
        with open(image_path, "rb") as f:
            image_data = f.read()
        mime_type = _get_mime_type(image_path)
        parts.append(
            types.Part(inline_data=types.Blob(data=image_data, mime_type=mime_type))
        )
    return parts


def _process_api_stream_response(stream, output_dir: str):
    """Processes the streaming response from the GenAI API, saving images and printing text."""
    file_index = 0
    for chunk in stream:
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue

        for part in chunk.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                timestamp = int(time.time())
                file_extension = mimetypes.guess_extension(part.inline_data.mime_type)
                file_name = os.path.join(
                    output_dir,
                    f"remixed_image_{timestamp}_{file_index}{file_extension}",
                )
                _save_binary_file(file_name, part.inline_data.data)
                file_index += 1
            elif part.text:
                print(part.text)


def _save_binary_file(file_name: str, data: bytes):
    """Saves binary data to a specified file."""
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"File saved to: {file_name}")


def _get_mime_type(file_path: str) -> str:
    """Guesses the MIME type of a file based on its extension."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for {file_path}")
    return mime_type


def main():
    parser = argparse.ArgumentParser(
        description="Remix images using Google Generative AI."
    )
    parser.add_argument(
        "-i",
        "--image",
        action="append",
        required=True,
        help="Paths to input images (1-5 images). Provide multiple -i flags for multiple images.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Optional prompt for remixing the images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save the remixed images.",
    )

    args = parser.parse_args()

    all_image_paths = args.image

    num_images = len(all_image_paths)
    if not (1 <= num_images <= 5):
        parser.error("Please provide between 1 and 5 input images using the -i flag.")

    # Determine the prompt
    final_prompt = args.prompt
    if final_prompt is None:
        if num_images == 1:
            final_prompt = "Turn this image into a professional quality studio shoot with better lighting and depth of field."
        else:
            final_prompt = "Combine the subjects of these images in a natural way, producing a new image."

    # Ensure output directory exists
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    remix_images(
        image_paths=all_image_paths,
        prompt=final_prompt,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()