#include "../include/relay.hpp"
#include "../include/utils.hpp"


	

Request::Request(int size) {

    buffer = (unsigned char*) malloc( size * sizeof(unsigned char));

}

Request& Request::address(uint8_t addr) {
    buffer[0] = addr;
    return (*this);
}

Request& Request::read_output(int start, int count) {
    buffer = (unsigned char*)realloc(buffer, 8 * sizeof(unsigned char));

    buffer[1] = 0x01;
    buffer[2] = (uint8_t)((uint16_t)start & 0xFF00);
    buffer[3] = (uint8_t)((uint16_t)start & 0x00FF);
    buffer[4] = (uint8_t)((uint16_t)count & 0xFF00);
    buffer[5] = (uint8_t)((uint16_t)count & 0x00FF);

    uint16_t crc = modbus_crc16(buffer, 6);

    buffer[6] = (uint8_t) (crc & 0x00FF);
    buffer[7] = (uint8_t) (crc >> 8);

    return (*this);
}

Request& Request::write_output(int channel, int value) {
    buffer = (unsigned char*)realloc(buffer, 8 * sizeof(unsigned char));
    
    buffer[1] = 0x05;
    buffer[2] = (uint8_t)((uint16_t)channel & 0xFF00);
    buffer[3] = (uint8_t)((uint16_t)channel & 0x00FF);
    buffer[4] = (uint8_t)((uint16_t)value >> 8);
    buffer[5] = (uint8_t)((uint16_t)value & 0x00FF);

    uint16_t crc = modbus_crc16(buffer, 6);

    buffer[6] = (uint8_t) (crc & 0x00FF);
    buffer[7] = (uint8_t) (crc >> 8);
    

    return (*this);
}

Request& Request::write_multi_output() {
    buffer[1] = 0x15;
    return (*this);
}

Request& Request::read_input(int start, int count) {
    buffer = (unsigned char*)realloc(buffer, 8 * sizeof(unsigned char));

    buffer[1] = 0x02;

    buffer[2] = (uint8_t)((uint16_t)start & 0xFF00);
    buffer[3] = (uint8_t)((uint16_t)start & 0x00FF);
    buffer[4] = (uint8_t)((uint16_t)count & 0xFF00);
    buffer[5] = (uint8_t)((uint16_t)count & 0x00FF);

    uint16_t crc = modbus_crc16(buffer, 6);

    buffer[6] = (uint8_t) (crc & 0x00FF);
    buffer[7] = (uint8_t) (crc >> 8);
    return (*this);
}

Request& Request::read_spec_reg() {
    buffer[1] = 0x03;
    return (*this);
}

Request& Request::write_spec_reg() {
    buffer[1] = 0x06;
    return (*this);
}

Request& Request::write_multi_spec_reg() {
    buffer[1] = 0x16;
    return (*this);
}


Relay::Relay(std::string path) {

    uart = open(path.c_str(), O_RDWR);
    usleep(100000);
    if (uart < 0) { perror("open"); return ; }

    
    tcgetattr(uart, &tty);
    cfmakeraw(&tty);
    
    tty.c_iflag &= ~(PARMRK | INPCK);
    tty.c_iflag |= IGNPAR;


   

    cfsetospeed(&tty, B9600);
    cfsetispeed(&tty, B9600);
    tty.c_cflag &= ~PARENB; 
    tty.c_cflag &= ~CSTOPB; 
    tty.c_cflag &= ~CSIZE;  
    tty.c_cflag |= CS8;    

    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 70;  

    tcsetattr(uart, TCSAFLUSH, &tty);

}

int Relay::hexstr_to_bytes(const char *hexstr, unsigned char *out, size_t max_len) {
    size_t count = 0;
    while (*hexstr && count < max_len) {
        unsigned int byte;
        if (sscanf(hexstr, "%2x", &byte) != 1)
            break;
        out[count++] = (unsigned char)byte;

        // Пропустить два символа и возможный пробел
        while (*hexstr && *hexstr != ' ') hexstr++;
        while (*hexstr == ' ') hexstr++;
    }
    return count;
}


void Relay::send(Request req) {
    
    tcflush(uart, TCIOFLUSH);

    write(uart, req.buffer, 8);
    tcdrain(uart);   
    
}


int Relay::receive(uint8_t** buffer) {

    int n = read(uart, *buffer, 15);

    return n;
}