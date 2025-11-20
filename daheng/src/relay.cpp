#include "../include/relay.hpp"



class Request {
    
    public:

        unsigned char* buffer;

        Request(int size) {

            buffer = (unsigned char*) malloc( size * sizeof(unsigned char));

        }

        Request& address(uint8_t addr) {
            buffer[0] = addr;
            return (*this);
        }

        Request& read_output() {
            buffer[1] = 0x01;
            return (*this);
        }

        Request& write_output() {
            buffer[1] = 0x05;
            return (*this);
        }

        Request& write_multi_output() {
            buffer[1] = 0x15;
            return (*this);
        }

        Request& read_input() {
            buffer[1] = 0x02;
            return (*this);
        }

        Request& read_spec_reg() {
            buffer[1] = 0x03;
            return (*this);
        }

        Request& write_spec_reg() {
            buffer[1] = 0x06;
            return (*this);
        }

        Request& write_multi_spec_reg() {
            buffer[1] = 0x16;
            return (*this);
        }

        

};

Relay::Relay(std::string path) {

    rxbuf = (unsigned char*) malloc( 15 * sizeof(unsigned char));
    uart = open(path.c_str(), O_RDWR | O_SYNC);

    tcgetattr(uart, &tty);
    cfmakeraw(&tty);

    tty.c_iflag &= ~(PARMRK | INPCK);
    tty.c_iflag |= IGNPAR;


   

    cfsetospeed(&tty, B9600);
    cfsetispeed(&tty, B9600);
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8; 

    tty.c_cflag &= ~(PARENB | PARODD);      
    tty.c_cflag &= ~CSTOPB;                


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

uint16_t Relay::modbus_crc16( const unsigned char *buf, unsigned int len )
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

void Relay::send(std::string data) {

    unsigned char txbuf[data.length()];

    int len = hexstr_to_bytes(data.c_str(), txbuf, data.length() - 2);
    
    uint16_t crc = modbus_crc16(txbuf, 6);
    
    txbuf[data.length() - 2] = crc & 0xFF; 
    txbuf[data.length() - 1] = (crc >> 8) & 0xFF;

    tcflush(uart, TCIOFLUSH); 
    tcflush(uart, TCIFLUSH);     
    write(uart, &txbuf, sizeof(txbuf));
    tcdrain(uart);   
    
}

void Relay::receive() {

    int n = read(uart, rxbuf, 15);

}