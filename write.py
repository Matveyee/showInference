import pymodbus.client as ModbusClient

client = ModbusClient.ModbusSerialClient(
    # method='rtu',
    port='/dev/ttyS1',
    baudrate=9600,
    timeout=3,
    parity='N',
    stopbits=1,
    bytesize=8
)

client.connect()

result = client.read_holding_registers(address=0, count=10)

if result.isError():
    print("Ошибка:", result)
else:
    print("Ответ контроллера:", result.registers)

# пример записи в один регистр
# client.write_register(address=0, value=123, unit=1)

# закрываем порт
client.close()