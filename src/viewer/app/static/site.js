$(document).on('click', '.confirm-delete', function () {
    return confirm('Are you sure you want to delete this?');
})

function world_bank_ajax_call() {
    var url = $("#WorldBankForm").attr("indicator-autocomplete-uri");
    var topic = $("#id_topic").val();
    var country = ""
    if (url.endsWith("/scm")) {
        country = $("#id_countries").val();
    } else {
        country = $("#id_country").val();
    }
    if (url.endsWith("/scmm")) {
        indicator_id = "#id_indicators"
    } else {
        indicator_id = "#id_indicator"
    }

    $("#loading").html('<img src="/static/giphy.webp" width="24px" alt="please wait... loading">')
    $.ajax({
        type: "GET",
        url: url + "?topic_id=" + topic + "&country=" + country,
        success: function (data) {
            $(indicator_id).html(data);
            $("#loading").html("");
        }
    });

};
