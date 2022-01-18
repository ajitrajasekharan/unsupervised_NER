echo "Shutting down servers"
pkill -9 python
sleep 5
echo "cleaning microservices directory"
rm -rf services labels models
