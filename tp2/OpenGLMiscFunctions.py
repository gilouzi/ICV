import cv2
import numpy as np

from OpenGL.GL import *

def loadBackgroundTexture(img):
    '''
        Função que carrega a textura do fundo da cena para a OpenGL
        
        Parâmetros:
        ----------
        img : numpy.ndarray
            Imagem a ser usada como textura
            
        Retorno:
        -------
        int
            Id da textura criado pela OpenGL
    '''
    
    # Criando um id para a textura e habilitando
    background_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, background_id)
    
    # Convertendo a imagem de BGR para RGB e realizando um 'flip'
    background = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    background = cv2.flip(background, 0)
    
    # Convertendo a imagem para uma string
    height, width, channels = background.shape
    background = np.fromstring(background.tostring(), dtype=background.dtype, count = height * width * channels)    
    background.shape = (height, width, channels)

    # Criando a textura na OpenGL
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, background)
    
    # Retornando o id
    return background_id