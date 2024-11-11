# Llamada a la API de OpenAI para generar la imagen
response = client.images.generate()
model = "dall-e-3",
prompt = user_input,
n = 1,
size = "1024x1024"
image_url = response.data[0].url

# Descarga de la imagen
image_response = requests.get(image_url)
image = Image.open(BytesIO(image_response.content))

# Guardar la imagen
image_path = os.path.join(output_folder, 'imagen_generada.png')
image.save(image_path)

# Mostrar la imagen
st.image(image, caption='Imagen Generada')

# Mensaje de éxito
st.success('La imagen ha sido creada y guardada con éxito.')