var mUploadManager = function() {
    
    var ACCEPT_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
    var qu = {};

    function checkUpload(filelist) {
        console.log(filelist);
        var files = filelist.get(0).files;
        for (var i=0; i<files.length; i++ ) {
            var fobj = files[i]
            //window.classify.enqu(fobj);
            console.log(fobj);
        }
    }

    function enqueue(fobj) {
        if ( fobj instanceof File && $.inArray(fobj.type, ACCEPT_TYPES) ) {
            var rid = window.classify.md5( '' + [service, fobj.type, fobj.name, fobj.lastModified, fobj.size].join() );
            if ( !(rid in qu) ) {
                upload( rid, fobj );
                qu[rid] = {
                    file : fobj,
                    status : 'queued'
                };
            }
        }
    }

    function upload( rid, fobj ) {

        var table = document.getElementById("table_results");

        // Insert a row to display uploading item
        var row = table.insertRow(0);
        row.id = rid;
        var cell1 = row.insertCell(0);
        cell1.width = "420px";
        var cell2 = row.insertCell(1);
        cell1.innerHTML = '<img src="http://192.168.56.102/images/loading.gif"></img>';

        // Convert actions to service / model
        var type = '';
        var model = '';
        var action = $("input[name='service']:checked").val();
        if ( action=='/classify/inceptionv3' ) {
            type = 'CLASSIFY';
        } else if ( action=='/classify/darknet19' ) {
        } else if ( action=='/detection/yolo' ) {
            type = 'DETECT';
        } else if ( action=='/detection/tinyyolo' ) {
        } else if ( action=='/face/detect' ) {
            type = 'FACE';
        } else if ( action=='/face/recognize' ) {
        }

        console.log( ['upload', rid] );
        var reader = new FileReader();
        reader.onloadend = function() {
            var base64data = reader.result;
            base64data = base64data.split("base64,")[1];
            console.log( ['onloadend', base64data.length]);
            var requestObj = {
                "requests": [
                    {
                        "requestId" : rid,
                        "media" : {
                            "content" : base64data
                        },
                        "services" : [
                            {
                                "type" : type,
                                "model" : model,
                                "options" : {
                                    "resultsLimit" : 5,
                                },
                            }
                        ]
                    }
                ]
            };
            var postdata = JSON.stringify(requestObj);

            // Post image url or data to service
            $.ajax( {

                url: '$action',
                type: 'POST',
                dataType : "json",
                contentType : "application/json; charset=utf-8",
                data : postdata,
                processData : false,
                error : function ( response ) {
                    console.log( response );
                },
                success : function ( responses ) {
                    console.log('ajax_success');
                    var err_no = 0;

                    if ( !responses.hasOwnProperty('responses') ) {
                        console.log('Corrupted responses');
                        console.log(responses);
                        return;
                    }

                    console.log( responses.responses );

                    for ( var i=0; i < responses.responses.length; i++ ) {
                        var response = responses.responses[i];
                        console.log( response );
                        var rid = response.requestId;

                        var tr = $("#"+rid);
                        var cell1 = $( tr.find("td").get(0) );
                        var cell2 = $( tr.find("td").get(1) );
                        if ( response.hasOwnProperty('preview') ) {
                            cell1.html( '<img src="' + response.preview + '"></img>' );
                        }

                        if ( response.hasOwnProperty('detectEntities') ) {
                            console.log( response.detectEntities );
                            var htmlstr = '';
                            //htmlstr = htmlstr + 'returned in ' + response.result.exec_time + ' ms<br/><br/>';
                            for ( var j=0; j < response.detectEntities.length; j++ ) {
                                var entity = response.detectEntities[j];
                                htmlstr = htmlstr + entity.label + ' / ' + entity.score + '<br />';
                            }
                            cell2.html(htmlstr);
                        } else if ( response.hasOwnProperty('faceEntities') ) {
                            console.log( response.faceEntities );
                            var htmlstr = '';
                            //htmlstr = htmlstr + 'returned in ' + response.result.exec_time + ' ms<br/><br/>';
                            for ( var j=0; j < response.faceEntities.length; j++ ) {
                                var entity = response.faceEntities[j];
                                htmlstr = htmlstr + entity.label + ' / ' + entity.score + '<br />';
                            }
                            cell2.html(htmlstr);
                        } else if ( response.hasOwnProperty('predictions') ) {
                            console.log( response.predictions );
                            var htmlstr = '';
                            //htmlstr = htmlstr + 'returned in ' + response.result.exec_time + ' ms<br/><br/>';
                            for ( var j=0; j < response.predictions.length; j++ ) {
                                var prediction = response.predictions[j];
                                htmlstr = htmlstr + prediction.class + ' / ' + prediction.score + '<br />';
                            }
                            cell2.html(htmlstr);
                        }
                    }
                }
            } );
        }
        reader.readAsDataURL(fobj); 
    },
    
    function something() {
        console.log('something');
    }

    return {
        checkUpload : checkUpload,
        something : something
    }

}();