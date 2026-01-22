        size_t data_offset = reinterpret_cast<size_t>(pointer);
        data_pointer = m_array_buffer->offset_data(data_offset);
    }
    tex_coord_pointer = { .size = size, .type = type, .normalize = false, .stride = stride, .pointer = data_pointer };
}

void GLContext::gl_vertex(GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
    APPEND_TO_CALL_LIST_AND_RETURN_IF_NEEDED(gl_vertex, x, y, z, w);

    GPU::Vertex vertex;

    vertex.position = { x, y, z, w };
    vertex.color = m_current_vertex_color;
    for (size_t i = 0; i < m_device_info.num_texture_units; ++i)
        vertex.tex_coords[i] = m_current_vertex_tex_coord[i];
    vertex.normal = m_current_vertex_normal;
