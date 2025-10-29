#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <string.h>
#include <stdlib.h>
#include <sstream>
#include <bitset>
#include <iostream>
#include "../include/modbus.hpp"
uint16_t modbus_crc16( const unsigned char *buf, unsigned int len )
{
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

int hexstr_to_bytes(const char *hexstr, unsigned char *out, size_t max_len) {
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



ModBus::ModBus() {}

ModBus::ModBus(const char* path) {
    self_fd = open(path, O_RDWR | O_SYNC);
}

void ModBus::init(const char* path , int speed, int parity, int size, int stop_bit) {
    
    tcgetattr(self_fd, &self_termios);
    cfmakeraw(&self_termios);

    self_termios.c_iflag &= ~(PARMRK | INPCK);
    self_termios.c_iflag |= IGNPAR;

    setSpeed(speed);

    self_termios.c_cflag &= ~(PARENB | PARODD);      
    self_termios.c_cflag &= ~CSTOPB;                 
    tcflush(self_fd, TCIFLUSH);
    tcsetattr(self_fd, TCSAFLUSH, &self_termios);
    
}
void ModBus::send(char* input) {
    unsigned char txbuf[8];
    int len = hexstr_to_bytes(input, txbuf, 6);
    uint16_t crc = modbus_crc16(txbuf, 6);
    txbuf[6] = crc;
    uint16_t crc2 = modbus_crc16(txbuf, 7);
    txbuf[7] = crc2;

    tcflush(self_fd, TCIFLUSH);     
    write(self_fd, &txbuf, sizeof(txbuf));
    tcdrain(self_fd);                
}

void ModBus::receive(char* rxbuf) {
    tcflush(ModBus::self_fd, TCIFLUSH);
    tcflush(ModBus::self_fd, TCIOFLUSH); 
// sleep(2);            
    int n = read(ModBus::self_fd, rxbuf, sizeof(rxbuf));
}
void ModBus::setSpeed(int speed) {


    if (speed == 300) {
        cfsetospeed(&self_termios, B300);
        cfsetispeed(&self_termios, B300);
    } else if (speed == 600 ) {
        cfsetospeed(&self_termios, B600);
        cfsetispeed(&self_termios, B600);
    } else if (speed == 1200) {
        cfsetospeed(&self_termios, B1200);
        cfsetispeed(&self_termios, B1200);
    } else if (speed == 2400) { 
        cfsetospeed(&self_termios, B2400);
        cfsetispeed(&self_termios, B2400);
    } else if (speed == 4800) {
        cfsetospeed(&self_termios, B9600);
        cfsetispeed(&self_termios, B9600);
    } else if (speed == 192000) {
        cfsetospeed(&self_termios, B19200);
        cfsetispeed(&self_termios, B19200);
    } else if (speed == 38400) { 
        cfsetospeed(&self_termios, B38400);
        cfsetispeed(&self_termios, B38400);
    } else if (speed == 57600) {
        cfsetospeed(&self_termios, B57600);
        cfsetispeed(&self_termios, B57600);
    } else if (speed == 115200) {
        cfsetospeed(&self_termios, B115200);
        cfsetispeed(&self_termios, B115200);
    } else {
        std::cout << "Incorrect baud rate" << std::endl;
    }
    
}
