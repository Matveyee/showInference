#include "../include/utils.hpp"


void log(std::string message) {
    std::cout << "DEBUG: " << message << std::endl;
}

uint16_t modbus_crc16( const unsigned char *buf, unsigned int len ) {
	uint16_t crc = 0xFFFF;
	unsigned int i = 0;
	char bit = 0;

	for( i = 0; i < len; i++ )
	{
		crc ^= buf[i];

		for( bit = 0; bit < 8; bit++ )
		{
			if( crc & 0x0001 )
			{
				crc >>= 1;
				crc ^= 0xA001;
			}
			else
			{
				crc >>= 1;
			}
		}
	}
    return crc;
}

drmModeRes* res;
drmModeConnector* conn;
uint32_t conn_id;
drmModeCrtc *old_crtc = nullptr;
uint8_t* map = nullptr;
uint32_t fb_id;
uint32_t handle;
uint32_t pitch;
uint64_t size;
struct drm_mode_fb_cmd fb = {};



Vec::Vec(int p_x, int p_y) : x(p_x), y(p_y) {}
Vec::Vec() {}

void Vec::init(int p_x, int p_y) {
    x = p_x;
    y = p_y;
}

int Vec::getx() {
    return x;
}

int Vec::gety() {
    return y;
}
Vec& Vec::operator += (Vec& other) {
    x += other.getx();
    y += other.gety();
    return *this;
}
void Vec::print() {
    std::cout << "( " << x << "," << y << ")";
}



Vec operator +(Vec& first, Vec& second) {

    Vec vec(first.getx() + second.getx(), first.gety() + second.gety());
    return vec;

}

Projection::Projection() {}

Projection::Projection(uint8_t* data_ptr, int offset_x, int offset_y, int source_w, int source_h, int width, int height) : data(data_ptr), offsetX(offset_x), offsetY(offset_y), w(width), h(height), sourceW(source_w), sourceH(source_h) {}

uint8_t Projection::operator [](int index) {   
    
    return data[(offsetY + index / w) * sourceW + offsetX + index % w];
}
void Projection::init(uint8_t* data_ptr , int offset_x, int offset_y, int source_w, int source_h, int width, int height) {
    data = data_ptr;
    offsetX = offset_x;
    offsetY = offset_y;
    w = width;
    h = height;
    sourceW = source_w;
    sourceH = source_h;
}
uint8_t Projection::get(int x, int y) {
    if (x > w || y > h) {
        return 0;
    } else {
        return (*this)[x + y * w];
    }
    
}

uint8_t Projection::get(Vec vec) {
    return get(vec.getx(), vec.gety());
}

int Projection::getx() {
    return offsetX;
}

int Projection::gety() {
    return offsetY;
}

int Projection::getW() {
    return w;
}

int Projection::getH() {
    return h;
}


void draw_line(uint8_t* map, int x, int y, int x1, int y1, uint32_t pitch) {
    uint32_t pixel = (0xFF << 24) | (255 << 16) | (0 << 8) | 0;
    if (x == x1) {
        for (int i = y; i <= y1; i++) {
            ((uint32_t*)(map + i * pitch))[x] = pixel;
        }
    } else if (y == y1) {
        for (int i = x; i <= x1; i++) {
            ((uint32_t*)(map + y * pitch))[i] = pixel;
        }
    }
    
}

void draw_rect(uint8_t* map, int x, int y, int x1, int y1, uint32_t pitch) {
    draw_line(map,x,y,x1,y, pitch);
    draw_line(map,x,y1,x1,y1, pitch);
    draw_line(map,x,y,x,y1, pitch);
    draw_line(map,x1,y,x1,y1, pitch);
}






void specifyVectors(Vec& R, Vec& G1, Vec& G2, Vec& B, Vec r) {
    
    if ( (r.getx() % 2 == 0) && (r.gety() % 2 == 0)) {

        R.init(0,0);
        G1.init(1,0);
        G2.init(0,1);
        B.init(1,1);
    }else if ((r.getx() % 2 != 0) && (r.gety() % 2 == 0)) {

        G1.init(0,0);
        R.init(0,1);
        B.init(1,0);
        G2.init(1,1);
    }else if ((r.getx() % 2 == 0) && (r.gety() % 2 != 0)) {

        G2.init(0,0);
        B.init(1,0);
        R.init(0,1);
        G1.init(1,1);
    } else {

        B.init(0,0);
        G2.init(1,0);
        G1.init(0,1);
        R.init(1,1);
    }


}
uint8_t* toRGB(Projection input) {
//    debug("to rgb entered");
   // debug("entered to rgb");
    uint8_t* rgbData = (uint8_t*)calloc(  input.getW() * input.getH() * 3, sizeof(uint8_t));
   // debug("mem allocated");
    Vec r;
    Vec r0;
    Vec R;
    Vec G1;
    Vec G2;
    Vec B;
    r.init(0,0);
    r0.init(input.getx(), input.gety());
   // debug("before cycle");
    for (int i = 0; i < (input.getW() * input.getH()); i++ ) {
        Vec d = r0 + r;
        specifyVectors(R, G1, G2, B, d);
        rgbData[i * 3 + 0] = input.get(r + R);
        rgbData[i * 3 + 1] = ( input.get(r + G1) + input.get(r + G2) ) / 2;
        rgbData[i * 3 + 2] = input.get(r + B);
        

        Vec delta;
        if ( (i + 1) % input.getW() == 0 ) {
            delta.init(1 - input.getW(), 1);
        } else {
            delta.init(1,0);
        }
        r += delta;
    }
   // debug("before return");
   return rgbData;
}



void drawPicture(Projection* proj, uint8_t* data) {
    log("Draw picture entered");
     for (int y = 0; y < 640; ++y) {
        for (int x = 0; x < 640; ++x) {
            int src_offset = y * 640 * 3  + x * 3;
            
            uint8_t r = data[src_offset + 0];
            uint8_t g = data[src_offset + 1];
            uint8_t b = data[src_offset + 2];

            uint32_t pixel = (0xFF << 24) | (r << 16) | (g << 8) | b;
          
            ((uint32_t*)(map + (y) * pitch))[x] = pixel;
        }

    }
    log("drawPicture exit");
}


void drm_init(int drm_fd) {

    log("Drm opened");
    if (drm_fd < 0) {
        perror("open");
        return;
    }

    res = drmModeGetResources(drm_fd);
    conn = nullptr;
    drmModeModeInfo mode;
    conn_id = 0;
    for (int i = 0; i < res->count_connectors; ++i) {
        conn = drmModeGetConnector(drm_fd, res->connectors[i]);
        if (conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
            mode = conn->modes[0];
            conn_id = conn->connector_id;
            break;
        }
        drmModeFreeConnector(conn);
    }
    log("Drm connector");

    if (!conn_id) {
        std::cerr << "No connected display found\n";
        return;
    }

    drmModeEncoder* enc = drmModeGetEncoder(drm_fd, conn->encoder_id);
    uint32_t crtc_id = enc->crtc_id;
    old_crtc = drmModeGetCrtc(drm_fd, crtc_id);
    log("Drm encoder");
    struct drm_mode_create_dumb create = {};
    create.width = mode.hdisplay;
    create.height = mode.vdisplay;
    create.bpp = 32;
    ioctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &create);
    handle = create.handle;
    pitch = create.pitch;
    size = create.size;

    struct drm_mode_map_dumb map_dumb = {};
    map_dumb.handle = handle;
    ioctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &map_dumb);
    map = (uint8_t*)mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, drm_fd, map_dumb.offset);
    log("Drm mmap");
    
    fb.width = mode.hdisplay;
    fb.height = mode.vdisplay;
    fb.pitch = pitch;
    fb.bpp = 32;
    fb.depth = 24;
    fb.handle = handle;
    drmModeAddFB(drm_fd, fb.width, fb.height, fb.depth, fb.bpp, pitch, handle, &fb_id);

    drmModeSetCrtc(drm_fd, crtc_id, fb_id, 0, 0, &conn_id, 1, &mode);

} 

void drm_destroy(int drm_fd) {

    if (old_crtc) {
        drmModeSetCrtc(drm_fd, old_crtc->crtc_id, old_crtc->buffer_id,
                       old_crtc->x, old_crtc->y,
                       &conn_id, 1, &old_crtc->mode);
        drmModeFreeCrtc(old_crtc);
    }

    munmap(map, size);
    drmModeRmFB(drm_fd, fb_id);

    struct drm_mode_destroy_dumb destroy = {0};
    destroy.handle = handle;
    drmIoctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);

    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    close(drm_fd);

}
