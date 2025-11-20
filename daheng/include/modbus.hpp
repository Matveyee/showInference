#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <string.h>
#include <stdlib.h>
#include <sstream>
#include <bitset>
#include <iostream>

uint16_t modbus_crc16( const unsigned char *buf, unsigned int len );

int hexstr_to_bytes(const char *hexstr, unsigned char *out, size_t max_len);


class ModBus {

    public:

        ModBus() {}

        ModBus(const char* path);

        void init(const char* path , int speed, int parity, int size, int stop_bit);
        
        void send(char* input);

        void receive(char* rxbuf);

    private:

        int self_fd;
        struct termios self_termios;
        
        void setSpeed(int speed);
};