import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr
import numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGGEncoder(nn.Module):
    def __init__(self, weight_path):
        super().__init__()

        state_dict = torch.load(weight_path, map_location=device)

        encoder_layers = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(inplace=True),
        )

        try:
            encoder_layers.load_state_dict(state_dict, strict=False)
        except:
            encoder_state = {}
            layer_mapping = {'0': 0, '2': 2, '5': 5, '9': 9, '12': 12,
                             '16': 16, '19': 19, '22': 22, '25': 25, '29': 29}
            for state_key, state_value in state_dict.items():
                layer_num = state_key.split('.')[0]
                if layer_num in layer_mapping:
                    new_key = f"{layer_mapping[layer_num]}.{state_key.split('.')[1]}"
                    encoder_state[new_key] = state_value
            encoder_layers.load_state_dict(encoder_state, strict=False)

        self.encoder = encoder_layers
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, weight_path):
        super().__init__()

        decoder_layers = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

        if Path(weight_path).exists():
            state_dict = torch.load(weight_path, map_location=device)
            decoder_layers.load_state_dict(state_dict)

        self.decoder = decoder_layers

    def forward(self, x):
        return self.decoder(x)


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)
                       ) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def load_image(img, size=512):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype('uint8'))
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)


def save_image(tensor):
    img = tensor.cpu().clone().detach().squeeze(0)
    img = torch.clamp(img, 0, 1)
    img = img.numpy().transpose(1, 2, 0)
    return (img * 255).astype(np.uint8)


vgg_path = "models/vgg_normalized.pth"
decoder_path = "models/decoder.pth"

encoder = VGGEncoder(vgg_path).to(device).eval()
decoder = Decoder(decoder_path).to(device).eval()


def style_transfer(content_img, style_img, alpha=1.0):
    if content_img is None or style_img is None:
        return np.zeros((512, 512, 3), dtype=np.uint8)

    try:
        content = load_image(content_img, 512).to(device)
        style = load_image(style_img, 512).to(device)

        with torch.no_grad():
            content_feat = encoder(content)
            style_feat = encoder(style)
            t = adaptive_instance_normalization(content_feat, style_feat)
            t = alpha * t + (1 - alpha) * content_feat
            output = decoder(t)

        result = save_image(output)
        return result

    except Exception as e:
        print(f"Error: {e}")
        return np.zeros((512, 512, 3), dtype=np.uint8)


custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%) !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

#title-container {
    text-align: center;
    padding: 2rem 0 1rem 0;
}

#title {
    color: #00d9ff;
    font-size: 2.8em;
    font-weight: 700;
    margin-bottom: 0.3rem;
    text-shadow: 0 0 30px rgba(0, 217, 255, 0.4);
    letter-spacing: -0.02em;
}

#subtitle {
    color: #8b9dc3;
    font-size: 1.15em;
    font-weight: 400;
    margin-bottom: 0.5rem;
}

#credit {
    color: #6b7c9d;
    font-size: 0.9em;
    font-style: italic;
    margin-top: 0.5rem;
}

.image-upload-container {
    display: flex;
    gap: 1rem;
    margin: 2rem 0;
}

.input-column, .output-column {
    flex: 1;
}

.image-container {
    border-radius: 12px !important;
    border: 1px solid rgba(0, 217, 255, 0.25) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
    overflow: hidden !important;
}

.control-container {
    background: rgba(26, 31, 58, 0.5);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    border: 1px solid rgba(0, 217, 255, 0.15);
}

.slider-container {
    margin-bottom: 1rem;
}

button.generate-btn {
    background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%) !important;
    border: 1px solid rgba(0, 217, 255, 0.3) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.1em !important;
    padding: 0.8rem 2rem !important;
    border-radius: 8px !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 102, 204, 0.3) !important;
    cursor: pointer !important;
}

button.generate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(0, 217, 255, 0.5) !important;
}

.info-box {
    background: rgba(26, 31, 58, 0.5);
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 2rem;
    border: 1px solid rgba(0, 217, 255, 0.15);
    text-align: center;
    color: #8b9dc3;
}

.info-text {
    font-size: 0.95em;
    line-height: 1.6;
    margin: 0.3rem 0;
}

label {
    color: #c5d1e8 !important;
    font-weight: 500 !important;
    font-size: 1em !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Default(primary_hue="blue", secondary_hue="cyan")) as demo:
    gr.HTML("""
    <div id="title-container">
        <h1 id="title">Neural Style Transfer Web App</h1>
        <p id="subtitle">Arbitrary style transfer with AdaIN algorithm</p>
        <p id="credit">Developed by Andy Ting</p>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            content_input = gr.Image(
                label="Content Image",
                type="numpy",
                elem_classes="image-container",
                format="png"
            )
            style_input = gr.Image(
                label="Style Image",
                type="numpy",
                elem_classes="image-container",
                format="png"
            )

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Stylized Result",
                elem_classes="image-container",
                format="png"
            )

    with gr.Row():
        with gr.Column():
            alpha_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.05,
                label="Style Strength",
                info="Adjust the intensity of style transfer (0 = content only, 1 = full style)"
            )

    with gr.Row():
        transfer_btn = gr.Button(
            "Generate Style Transfer",
            variant="primary",
            size="lg",
            elem_classes="generate-btn"
        )

    gr.HTML("""
    <div class="info-box">
        <p class="info-text"><strong>Instructions:</strong> Upload a content image and a style image, then click generate.</p>
        <p class="info-text">Images will be resized to 512×512 for optimal performance.</p>
        <p class="info-text">GPU acceleration can be enabled for fast processing.</p>
    </div>
    """)

    transfer_btn.click(
        fn=style_transfer,
        inputs=[content_input, style_input, alpha_slider],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch()
