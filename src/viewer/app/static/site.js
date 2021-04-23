$(document).on('click', '.confirm-delete', function() {
    return confirm('Are you sure you want to delete this?');
})

$(document).on('change', 'select#id_topic', function () {
    var url = $("#SCSMForm").attr("indicator-autocomplete-uri");  
    var id = $(this).val();
    var country = $("#id_country").val();
    
    $("#loading").html('<img src="/static/giphy.webp" width="24px" alt="please wait... loading">')
    $.ajax({
       type: "GET",
       url: url + "?topic_id="+ id+"&country="+country,
       success: function (data) {   
          $("#id_indicator").html(data);  
          $("#loading").html("");
       },
       error: function(xhr) {
            alert("Unable to load available datasets: "+xhr.statusText);
       }
    });
  
  });
  