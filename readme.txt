Essential Services:
    nginx
    uwsgi-plugin-python

Python Tornado Server:
    - tornado must be installed with 'sudo -H' for systemd to access

    - /etc/systemd/system/pytor.service

        [Unit]
        Description=Python Tornado Server

        [Service]
        Type=simple
        ExecStart=/usr/bin/pytor

        [Install]
        WantedBy=multi-user.target

    - /usr/bin/pytor

        #!/bin/bash

        python /usr/bin/pytor-server

uWSGI:
    curl http://uwsgi.it/install | bash -s cgi [absolute_path]

    http://raspberrywebserver.com/cgiscripting/setting-up-nginx-and-uwsgi-for-cgi-scripting.html
    
        [uwsgi]
        plugins = cgi
        socket = 127.0.0.1:3031
        chdir = /usr/lib/cgi-bin/
        module = pyindex
        cgi=/cgi-bin=/usr/lib/cgi-bin/
        cgi-helper =.py=python
    
    http://uwsgi-docs.readthedocs.io/en/latest/Systemd.html

        [Unit]
        Description=uWSGI Local Server
        After=syslog.target

        [Service]
        ExecStart=/usr/bin/uwsgi --ini /etc/uwsgi/uwsgi_config.ini
        # Requires systemd version 211 or newer
        RuntimeDirectory=uwsgi
        Restart=always
        KillSignal=SIGQUIT
        Type=notify
        StandardError=syslog
        NotifyAccess=all

        [Install]
        WantedBy=multi-user.target
    
Resources:
    https://www.digitalocean.com/community/tutorials/how-to-set-up-uwsgi-and-nginx-to-serve-python-apps-on-centos-7
    http://uwsgi-docs.readthedocs.io/en/latest/Upstart.html
    https://gist.github.com/didip/802576
    https://stackoverflow.com/questions/14749655/setting-up-a-tornado-web-service-in-production-with-nginx-reverse-proxy
    
